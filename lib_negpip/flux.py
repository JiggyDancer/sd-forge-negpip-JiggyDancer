from functools import wraps
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from backend.sampling import condition, sampling_function
from modules import shared

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

def patch_flux_negpip(cls: "NegPiP", *, unpatch=False):
    if len(cls._patched) < 3:
        cls._patched.append(False)
        
    if unpatch != cls._patched[2]:
        return

    cls._patched[2] = not cls._patched[2]

    model = shared.sd_model
    dit = model.forge_objects.unet.model.diffusion_model
    
    _hook_flux_learned_conditioning(model, unpatch)
    _hook_flux_dit_forward(dit, unpatch)
    _hook_flux_compile_conditions(unpatch)

def _hook_flux_learned_conditioning(model, remove: bool):
    if remove:
        if hasattr(model, "orig_flux_forward"):
            model.get_learned_conditioning = model.orig_flux_forward
            del model.orig_flux_forward
        return

    model.orig_flux_forward = model.get_learned_conditioning
    engine = getattr(model, "text_processing_engine_flux", getattr(model, "text_processing_engine", None))

    @torch.inference_mode()
    @wraps(model.orig_flux_forward)
    def negpip_flux_conditioning(prompt):
        conds = model.orig_flux_forward(prompt)
        
        crossattn = []
        negpip_mask = []
        _count = 0

        iterable_conds = conds if isinstance(conds, list) else [conds]

        for line, cond in zip(prompt, iterable_conds):
            if isinstance(cond, dict):
                cond_data = cond.get("crossattn", cond.get("txt", None))
            elif isinstance(cond, torch.Tensor):
                cond_data = cond
            else:
                cond_data = None

            if cond_data is None:
                continue

            cond_data = cond_data.reshape(-1, cond_data.shape[-1])
            
            mask = _build_flux_negpip_mask(
                engine,
                line,
                cond_data.shape[0],
                cond_data.device,
                cond_data.dtype,
            )

            _count += int((mask < 0).sum())
            crossattn.append(cond_data * mask.unsqueeze(-1).to(cond_data))
            negpip_mask.append(mask.unsqueeze(-1).to(cond_data))

        if _count > 0:
            key = "Negative" if prompt.is_negative_prompt else "Positive"
            print(f"NegPiP Flux Enable ({key}: {_count})")

        result = conds[0] if isinstance(conds, list) else conds
        
        if isinstance(result, dict):
            if "crossattn" in result:
                result["crossattn"] = torch.stack(crossattn, dim=0)
            elif "txt" in result:
                result["txt"] = torch.stack(crossattn, dim=0)
            result["c_negpip_mask"] = torch.stack(negpip_mask, dim=0)
            return [result]
        else:
            return [{
                "crossattn": torch.stack(crossattn, dim=0),
                "c_negpip_mask": torch.stack(negpip_mask, dim=0),
            }]

    model.get_learned_conditioning = negpip_flux_conditioning

def _build_flux_negpip_mask(engine, line: str, token_length: int, device, dtype):
    if not engine:
        return torch.ones(token_length, device=device, dtype=dtype)
        
    chunks = engine.tokenize_line(line)
    multipliers = []
    
    for chunk in chunks:
        multipliers.extend(getattr(chunk, "multipliers", getattr(chunk, "t5_multipliers", [])))

    if not multipliers:
        return torch.ones(token_length, device=device, dtype=dtype)

    weights = torch.tensor(multipliers, device=device, dtype=dtype)
    ones = torch.ones_like(weights)
    
    mask = torch.where(weights < 0, weights * 0.5, ones)

    if mask.shape[0] < token_length:
        mask = F.pad(mask, (0, token_length - mask.shape[0]), value=1.0)
    elif mask.shape[0] > token_length:
        mask = mask[:token_length]

    return mask

def _hook_flux_dit_forward(dit, remove: bool):
    if remove:
        if hasattr(dit, "orig_flux_forward"):
            if getattr(dit.forward, "_negpip", False):
                dit.forward = dit.orig_flux_forward
            del dit.orig_flux_forward
        return

    dit.orig_flux_forward = dit.forward

    @torch.inference_mode()
    @wraps(dit.orig_flux_forward)
    def negpip_forward(*args, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        
        # Safely pop the mask out of the kwargs (which come from Forge's extra_conds)
        negpip_mask = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
            # Rebuild the dict to avoid modifying defaults or shared memory references
            if transformer_options is None:
                transformer_options = {}
            else:
                transformer_options = dict(transformer_options)
                
            transformer_options["negpip_mask"] = negpip_mask
            
        kwargs["transformer_options"] = transformer_options
        return dit.orig_flux_forward(*args, **kwargs)

    negpip_forward._negpip = True
    dit.forward = negpip_forward

def _hook_flux_compile_conditions(remove: bool):
    if remove:
        if hasattr(condition, "orig_flux_forward"):
            condition.compile_conditions = condition.orig_flux_forward
            sampling_function.compile_conditions = condition.orig_flux_forward
            del condition.orig_flux_forward
        return

    condition.orig_flux_forward = condition.compile_conditions

    @wraps(condition.orig_flux_forward)
    def compile_conditions(cond):
        if cond is None:
            return None

        if isinstance(cond, dict) and "c_negpip_mask" in cond:
            if "crossattn" in cond and "vector" not in cond:
                cross_attn = cond["crossattn"]
                model_conds = {"c_crossattn": condition.ConditionCrossAttn(cross_attn)}
                model_conds["c_negpip_mask"] = condition.Condition(cond["c_negpip_mask"])
                return [dict(cross_attn=cross_attn, model_conds=model_conds)]
            
            compiled = condition.orig_flux_forward(cond)
            for c in compiled:
                if isinstance(c, dict) and "model_conds" in c:
                    c["model_conds"]["c_negpip_mask"] = condition.Condition(cond["c_negpip_mask"])
            return compiled

        return condition.orig_flux_forward(cond)

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions
