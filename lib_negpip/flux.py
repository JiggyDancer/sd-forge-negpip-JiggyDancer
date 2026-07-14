from functools import wraps
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

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
        
        _count = 0

        def process_tensor(txt_tensor):
            nonlocal _count
            if txt_tensor.ndim == 2:
                txt_tensor = txt_tensor.unsqueeze(0)
                
            b, l, d = txt_tensor.shape
            out_txt = []
            out_mask = []
            
            for i in range(b):
                line = prompt[i] if i < len(prompt) else prompt[-1]
                mask = _build_flux_negpip_mask(engine, line, l, txt_tensor.device, txt_tensor.dtype)
                _count += int((mask < 0).sum())
                
                out_txt.append(txt_tensor[i] * mask.unsqueeze(-1))
                out_mask.append(mask.unsqueeze(-1))
                
            return torch.stack(out_txt, dim=0), torch.stack(out_mask, dim=0)

        if isinstance(conds, dict):
            txt_key = "crossattn" if "crossattn" in conds else "txt"
            if txt_key in conds and conds[txt_key] is not None:
                new_txt, mask = process_tensor(conds[txt_key])
                conds[txt_key] = new_txt
                conds["c_negpip_mask"] = mask
            
            if _count > 0:
                print(f"NegPiP Flux Enable (Tokens: {_count})")
            return conds

        elif isinstance(conds, list):
            new_conds = []
            for cond in conds:
                if isinstance(cond, dict):
                    txt_key = "crossattn" if "crossattn" in cond else "txt"
                    if txt_key in cond and cond[txt_key] is not None:
                        new_txt, mask = process_tensor(cond[txt_key])
                        cond[txt_key] = new_txt
                        cond["c_negpip_mask"] = mask
                    new_conds.append(cond)
                elif isinstance(cond, torch.Tensor):
                    new_txt, mask = process_tensor(cond)
                    new_conds.append({
                        "crossattn": new_txt,
                        "c_negpip_mask": mask
                    })
                else:
                    new_conds.append(cond)
                    
            if _count > 0:
                print(f"NegPiP Flux Enable (Tokens: {_count})")
            return new_conds

        elif isinstance(conds, torch.Tensor):
            new_txt, mask = process_tensor(conds)
            if _count > 0:
                print(f"NegPiP Flux Enable (Tokens: {_count})")
            return [{"crossattn": new_txt, "c_negpip_mask": mask}]
            
        return conds

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
        negpip_mask = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
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

        # 1. Bypass Forge's compiler if we are dealing with our synthetic dictionary
        # This prevents the KeyError: 'vector' crash entirely.
        if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
            if "c_negpip_mask" in cond[0] and "crossattn" in cond[0] and "vector" not in cond[0] and "y" not in cond[0]:
                compiled = []
                for c_item in cond:
                    txt = c_item["crossattn"]
                    model_conds = {
                        "c_crossattn": condition.ConditionCrossAttn(txt),
                        "c_negpip_mask": condition.Condition(c_item["c_negpip_mask"])
                    }
                    compiled.append({"crossattn": txt, "model_conds": model_conds})
                return compiled

        # 2. For all native dictionaries, proceed with standard compilation
        compiled = condition.orig_flux_forward(cond)
        
        # 3. Inject our mask back into the native compiled object
        if isinstance(cond, dict) and "c_negpip_mask" in cond:
            for c in compiled:
                if isinstance(c, dict) and "model_conds" in c:
                    c["model_conds"]["c_negpip_mask"] = condition.Condition(cond["c_negpip_mask"])
        elif isinstance(cond, list):
            for i, c_item in enumerate(cond):
                if isinstance(c_item, dict) and "c_negpip_mask" in c_item:
                    if i < len(compiled) and isinstance(compiled[i], dict) and "model_conds" in compiled[i]:
                        compiled[i]["model_conds"]["c_negpip_mask"] = condition.Condition(c_item["c_negpip_mask"])

        return compiled

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions
