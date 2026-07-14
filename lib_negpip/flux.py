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
        conds = model.orig_flux_forward(prompt)
        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        _count = 0

        # Safely wrap raw tensor returns
        if isinstance(conds, dict):
            cond_list = [conds]
        elif isinstance(conds, list):
            cond_list = conds
        elif isinstance(conds, torch.Tensor):
            cond_list = [{"txt": conds}]
            conds = cond_list 
        else:
            return conds

        # Build the mask, but DO NOT alter the text tensors. Let Q and K stay intact.
        for i, c_item in enumerate(cond_list):
            if isinstance(c_item, dict):
                txt_tensor = c_item.get("crossattn", c_item.get("txt"))
                if txt_tensor is not None:
                    b, l, d = txt_tensor.shape if txt_tensor.ndim == 3 else (1, txt_tensor.shape[0], txt_tensor.shape[1])
                    line = prompt_list[i] if i < len(prompt_list) else prompt_list[-1]
                    
                    mask = _build_flux_negpip_mask(engine, line, l, txt_tensor.device, txt_tensor.dtype)
                    _count += int((mask < 0).sum())
                    
                    # Store mask as [batch, seq, 1] for later broadcasting
                    c_item["c_negpip_mask"] = mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1)

        if _count > 0:
            print(f"NegPiP Flux Enable (Tokens: {_count})")
            
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
    
    # We apply the pure weight (e.g., -1.5) directly to the mask
    mask = torch.where(weights < 0, weights, ones)

    if mask.shape[0] < token_length:
        mask = F.pad(mask, (0, token_length - mask.shape[0]), value=1.0)
    elif mask.shape[0] > token_length:
        mask = mask[:token_length]

    return mask

def _hook_flux_dit_forward(dit, remove: bool):
    if remove:
        if hasattr(dit, "orig_flux_forward"):
            dit.forward = dit.orig_flux_forward
            del dit.orig_flux_forward
        if hasattr(attention, "orig_attention_function"):
            attention.attention_function = attention.orig_attention_function
            del attention.orig_attention_function
        return

    dit.orig_flux_forward = dit.forward
    attention.orig_attention_function = attention.attention_function

    @torch.inference_mode()
    @wraps(attention.orig_attention_function)
    def negpip_flux_attention(q, k, v, heads, mask=None, *args, **kwargs):
        neg_mask = getattr(shared.state, "current_negpip_flux_mask", None)
        
        if neg_mask is not None:
            # v shape: [batch, seq_len, dim]
            b, s, d = v.shape
            b_m, txt_len, _ = neg_mask.shape
            
            working_mask = neg_mask.to(v.device, dtype=v.dtype)
            
            # Handle CFG expansion if user intentionally uses CFG > 1.0
            if b > b_m and b % b_m == 0:
                working_mask = working_mask.repeat(b // b_m, 1, 1)
                
            if s == txt_len:
                # DoubleStreamBlock isolated text attention
                v = v * working_mask
            elif s > txt_len:
                # SingleStreamBlock concatenated attention (text is leading)
                v_txt = v[:, :txt_len, :] * working_mask
                v_img = v[:, txt_len:, :]
                v = torch.cat([v_txt, v_img], dim=1)
                
        return attention.orig_attention_function(q, k, v, heads, mask, *args, **kwargs)

    @torch.inference_mode()
    @wraps(dit.orig_flux_forward)
    def negpip_forward(*args, **kwargs):
        negpip_mask = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
            shared.state.current_negpip_flux_mask = negpip_mask
        
        try:
            # Route attention through our surgical hook during the DiT pass
            attention.attention_function = negpip_flux_attention
            res = dit.orig_flux_forward(*args, **kwargs)
        finally:
            # Ensure the backend is safely restored
            attention.attention_function = attention.orig_attention_function
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

        # Always let Forge compile natively
        compiled = condition.orig_flux_forward(cond)
        
        # Gently slip the mask back in
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
