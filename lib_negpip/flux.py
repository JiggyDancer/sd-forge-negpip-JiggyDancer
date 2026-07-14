from functools import wraps
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from backend.sampling import condition, sampling_function
from backend import attention 
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
        # 1. Forge applies negative weights directly to embeddings natively here.
        conds = model.orig_flux_forward(prompt)
        
        _count = 0

        def process_tensor(txt_tensor):
            nonlocal _count
            if txt_tensor.ndim == 2:
                txt_tensor = txt_tensor.unsqueeze(0)
                
            b, l, d = txt_tensor.shape
            out_txt = []
            out_mask = []
            tokens_found = 0
            
            for i in range(b):
                line = prompt[i] if i < len(prompt) else prompt[-1]
                
                # Re-parse weights to find exactly which tokens Forge negated
                weights = _build_flux_negpip_mask(engine, line, l, txt_tensor.device, txt_tensor.dtype)
                
                # 2. Undo the negative embedding inversion! 
                # This restores Q and K to positive so they target the concept accurately with massive force.
                sign_flip = torch.where(weights < 0, torch.tensor(-1.0, device=weights.device, dtype=weights.dtype), torch.tensor(1.0, device=weights.device, dtype=weights.dtype))
                
                # Fix the embeddings
                fixed_txt = txt_tensor[i] * sign_flip.unsqueeze(-1)
                
                out_txt.append(fixed_txt)
                out_mask.append(sign_flip.unsqueeze(-1))
                tokens_found += int((weights < 0).sum())
                
            return torch.stack(out_txt, dim=0), torch.stack(out_mask, dim=0), tokens_found

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

    if weights.shape[0] < token_length:
        weights = F.pad(weights, (0, token_length - weights.shape[0]), value=1.0)
    elif weights.shape[0] > token_length:
        weights = weights[:token_length]

    return weights

def _hook_flux_dit_forward(dit, remove: bool):
    if remove:
        if hasattr(dit, "orig_flux_forward"):
            dit.forward = dit.orig_flux_forward
            del dit.orig_flux_forward
        if hasattr(attention, "orig_flux_negpip_attention"):
            attention.attention_function = attention.orig_flux_negpip_attention
            del attention.orig_flux_negpip_attention
        return

    dit.orig_flux_forward = dit.forward
    
    # 3. SPLIT ATTENTION HOOK
    if not hasattr(attention, "orig_flux_negpip_attention"):
        attention.orig_flux_negpip_attention = attention.attention_function

        @torch.inference_mode()
        @wraps(attention.orig_flux_negpip_attention)
        def negpip_flux_attention(q, k, v, heads, mask=None, *args, **kwargs):
            neg_mask = getattr(shared.state, "current_negpip_flux_mask", None)
            
            if neg_mask is not None:
                b, s, d = v.shape
                b_m, txt_len, _ = neg_mask.shape
                
                working_mask = neg_mask.to(v.device, dtype=v.dtype)
                
                if b > b_m and b % b_m == 0:
                    working_mask = working_mask.repeat(b // b_m, 1, 1)
                    
                if s > txt_len:
                    # SPLIT ATTENTION: This isolates the text stream from the image stream
                    q_txt, q_img = q[:, :txt_len, :], q[:, txt_len:, :]
                    
                    # 1. Text Query: Keep text embeddings 100% pure so it doesn't self-corrupt over 19 layers
                    out_txt = attention.orig_flux_negpip_attention(q_txt, k, v, heads, mask, *args, **kwargs)
                    
                    # 2. Image Query: Negate the Value vectors for the specific negative tokens
                    v_txt_neg = v[:, :txt_len, :] * working_mask
                    v_img = v[:, txt_len:, :]
                    v_neg = torch.cat([v_txt_neg, v_img], dim=1)
                    
                    out_img = attention.orig_flux_negpip_attention(q_img, k, v_neg, heads, mask, *args, **kwargs)
                    
                    return torch.cat([out_txt, out_img], dim=1)
                    
            return attention.orig_flux_negpip_attention(q, k, v, heads, mask, *args, **kwargs)

        attention.attention_function = negpip_flux_attention

    @torch.inference_mode()
    @wraps(dit.orig_flux_forward)
    def negpip_forward(*args, **kwargs):
        negpip_mask = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is None and "transformer_options" in kwargs:
            negpip_mask = kwargs["transformer_options"].pop("c_negpip_mask", None)

        if negpip_mask is not None:
            shared.state.current_negpip_flux_mask = negpip_mask
        
        try:
            res = dit.orig_flux_forward(*args, **kwargs)
        finally:
            if hasattr(shared.state, "current_negpip_flux_mask"):
                del shared.state.current_negpip_flux_mask
            
        return res

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

        compiled = condition.orig_flux_forward(cond)
        
        # Gently slip the mask down into the compiled dictionary
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
