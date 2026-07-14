from functools import wraps
from typing import TYPE_CHECKING, Optional
import re
import torch
import torch.nn.functional as F

from backend.sampling import condition, sampling_function
from backend import attention 
from modules import shared
from lib_negpip.utils import NEG_PATTERN

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

    @torch.inference_mode()
    @wraps(model.orig_flux_forward)
    def negpip_flux_conditioning(prompt):
        # Handle cases where prompt is a list of strings
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        clean_prompts = []
        batch_neg_data = []
        has_neg = False

        for p_text in prompts:
            # Handle prompt objects if Forge passed them instead of raw strings
            text = getattr(p_text, "text", p_text) if not isinstance(p_text, str) else p_text
            
            matches = re.findall(NEG_PATTERN, text)
            clean_text = text
            neg_data = []
            for m in matches:
                clean_text = clean_text.replace(m, "")
                core = m.strip("() ")
                parts = core.split(":")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    try:
                        weight = float(parts[1].strip())
                        neg_data.append((word, weight))
                        has_neg = True
                    except:
                        pass
            clean_prompts.append(clean_text.strip(" ,"))
            batch_neg_data.append(neg_data)

        # 1. Process individually if model requires per-prompt handling
        base_conds = []
        for p_text in clean_prompts:
            # Use original model call per item to ensure is_negative_prompt attribute persists
            out = model.orig_flux_forward([p_text])
            base_conds.extend(out if isinstance(out, list) else [out])
        
        if not has_neg:
            return base_conds

        _count = 0
        for i, c_item in enumerate(base_conds):
            neg_list = batch_neg_data[i]
            if not neg_list:
                continue
            
            txt_key = "crossattn" if "crossattn" in c_item else "txt"
            base_txt = c_item[txt_key]
            
            if base_txt.ndim == 2:
                base_txt = base_txt.unsqueeze(0)
            
            b, seq_len, dim = base_txt.shape
            
            # Keep original tokens, append negatives at the end
            mask_list = [torch.ones(seq_len, device=base_txt.device, dtype=base_txt.dtype)]
            txt_list = [base_txt]
            
            for (word, weight) in neg_list:
                # Encode negative concept separately
                w_cond = model.orig_flux_forward([word])
                w_item = w_cond[0] if isinstance(w_cond, list) else w_cond
                w_txt = w_item[txt_key]
                if w_txt.ndim == 2:
                    w_txt = w_txt.unsqueeze(0)
                
                txt_list.append(w_txt)
                mask_list.append(torch.full((w_txt.shape[1],), weight, device=base_txt.device, dtype=base_txt.dtype))
                _count += 1
                
            c_item[txt_key] = torch.cat(txt_list, dim=1)
            c_item["c_negpip_mask"] = torch.cat(mask_list, dim=0).unsqueeze(0).unsqueeze(-1).expand(b, -1, -1)

        if _count > 0:
            print(f"NegPiP Flux Enable (Isolated Targets: {_count})")

        return base_conds

    model.get_learned_conditioning = negpip_flux_conditioning

# Keep _hook_flux_dit_forward and _hook_flux_compile_conditions the same as before.
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
    
    if not hasattr(attention, "orig_flux_negpip_attention"):
        attention.orig_flux_negpip_attention = attention.attention_function

        @torch.inference_mode()
        @wraps(attention.orig_flux_negpip_attention)
        def negpip_flux_attention(q, k, v, heads, mask=None, *args, **kwargs):
            neg_mask = getattr(shared.state, "current_negpip_flux_mask", None)
            if neg_mask is not None:
                b, s, d = v.shape
                b_m, total_txt_len, _ = neg_mask.shape
                working_mask = neg_mask.to(v.device, dtype=v.dtype)
                if b > b_m and b % b_m == 0:
                    working_mask = working_mask.repeat(b // b_m, 1, 1)
                if s == total_txt_len:
                    v = v * working_mask
                elif s > total_txt_len:
                    v = torch.cat([v[:, :total_txt_len, :] * working_mask, v[:, total_txt_len:, :]], dim=1)
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
        if cond is None: return None
        compiled = condition.orig_flux_forward(cond)
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
