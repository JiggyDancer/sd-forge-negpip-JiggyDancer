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
        prompts = [prompt] if isinstance(prompt, str) else prompt

        clean_prompts = []
        batch_neg_data = []
        has_neg = False

        # 1. Manually parse negative words to protect the base LLM embeddings
        for p_text in prompts:
            matches = re.findall(NEG_PATTERN, p_text)
            clean_text = p_text
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

        # Helper to preserve SdConditioning metadata (prevents 'list' attribute crashes)
        def make_cond_obj(texts):
            if hasattr(prompt, "is_negative_prompt"):
                new_obj = type(prompt)(texts)
                for k, v in vars(prompt).items():
                    setattr(new_obj, k, v)
                return new_obj
            return texts

        # 2. Compile the clean base prompt (keeps the 'y' pooled vector pure)
        base_conds = model.orig_flux_forward(make_cond_obj(clean_prompts))
        
        if not has_neg:
            return base_conds

        is_dict_return = isinstance(base_conds, dict)
        cond_list = [base_conds] if is_dict_return else base_conds
        _count = 0
        
        # 3. Compile negative words separately and append them to the sequence
        for i, c_item in enumerate(cond_list):
            neg_list = batch_neg_data[i]
            if not neg_list:
                continue
            
            txt_key = "crossattn" if "crossattn" in c_item else "txt"
            if txt_key not in c_item:
                continue

            base_txt = c_item[txt_key]
            
            if base_txt.ndim == 2:
                base_txt = base_txt.unsqueeze(0)
            
            b, seq_len, dim = base_txt.shape
            
            # Mask for base prompt is all 1.0
            mask_list = [torch.ones(seq_len, device=base_txt.device, dtype=base_txt.dtype)]
            txt_list = [base_txt]

            # Handle Flux Positional IDs if they exist
            has_txt_ids = "txt_ids" in c_item
            if has_txt_ids:
                base_txt_ids = c_item["txt_ids"]
                txt_ids_list = [base_txt_ids]
            
            for (word, weight) in neg_list:
                word_cond = model.orig_flux_forward(make_cond_obj([word]))
                w_item = word_cond[0] if isinstance(word_cond, list) else word_cond
                w_txt = w_item[txt_key]
                if w_txt.ndim == 2:
                    w_txt = w_txt.unsqueeze(0)
                    
                w_seq_len = w_txt.shape[1]
                txt_list.append(w_txt)
                
                # Assign the negative weight to the mask for these appended tokens
                w_mask = torch.full((w_seq_len,), weight, device=base_txt.device, dtype=base_txt.dtype)
                mask_list.append(w_mask)

                # Append matching positional IDs
                if has_txt_ids and "txt_ids" in w_item:
                    txt_ids_list.append(w_item["txt_ids"])

                _count += 1
                
            final_txt = torch.cat(txt_list, dim=1)
            final_mask = torch.cat(mask_list, dim=0)
            
            c_item[txt_key] = final_txt

            # Concatenate txt_ids safely
            if has_txt_ids and len(txt_ids_list) == len(txt_list):
                tid_dim = 1 if base_txt_ids.ndim == 3 else 0
                c_item["txt_ids"] = torch.cat(txt_ids_list, dim=tid_dim)

            # Store mask to be retrieved in the DiT block
            c_item["c_negpip_mask"] = final_mask.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1)

        if _count > 0:
            print(f"NegPiP Flux Enable (Isolated Targets: {_count})")

        return base_conds if not is_dict_return else cond_list[0]

    model.get_learned_conditioning = negpip_flux_conditioning


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
    
    # Global hook for attention that manages scope via shared.state
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
                    # ONLY negate the targeted text values, leaving image values pure
                    v_txt = v[:, :total_txt_len, :] * working_mask
                    v_img = v[:, total_txt_len:, :]
                    v = torch.cat([v_txt, v_img], dim=1)
                    
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
