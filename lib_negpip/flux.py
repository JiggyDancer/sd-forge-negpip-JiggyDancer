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
            if txt_tensor.ndim == 2:
                txt_tensor = txt_tensor.unsqueeze(0)
                
            b, l, d = txt_tensor.shape
            out_txt = []
            out_mask = []
            tokens_found = 0
            
            for i in range(b):
                line = prompt[i] if i < len(prompt) else prompt[-1]
                mask = _build_flux_negpip_mask(engine, line, l, txt_tensor.device, txt_tensor.dtype)
                tokens_found += int((mask < 0).sum())
                
                out_txt.append(txt_tensor[i] * mask.unsqueeze(-1))
                out_mask.append(mask.unsqueeze(-1))
                
            return torch.stack(out_txt, dim=0), torch.stack(out_mask, dim=0), tokens_found

        # 1. Safely process the tensor natively without renaming keys to trick Forge
        if isinstance(conds, dict):
            for k in ["txt", "crossattn"]:
                if k in conds and conds[k] is not None:
                    conds[k], mask, c = process_tensor(conds[k])
                    conds["c_negpip_mask"] = mask
                    _count += c
                    break

        elif isinstance(conds, list):
            for c_item in conds:
                if isinstance(c_item, dict):
                    for k in ["txt", "crossattn"]:
                        if k in c_item and c_item[k] is not None:
                            c_item[k], mask, c = process_tensor(c_item[k])
                            c_item["c_negpip_mask"] = mask
                            _count += c
                            break

        elif isinstance(conds, torch.Tensor):
            new_txt, mask, c = process_tensor(conds)
            _count += c
            # Wrap as "txt" (Flux spec) instead of "crossattn" to prevent KeyError: 'vector'
            conds = {"txt": new_txt, "c_negpip_mask": mask}

        if _count > 0:
            key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
            print(f"NegPiP Flux Enable ({key}: {_count})")

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

        # 1. ALWAYS let Forge compile the object natively. No synthetic chunking bypasses.
        compiled = condition.orig_flux_forward(cond)
        
        # 2. Gently inject the mask directly into the compiled model_conds object
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
