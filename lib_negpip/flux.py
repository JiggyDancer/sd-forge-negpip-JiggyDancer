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

        # 1. Parse negative words out of the prompt
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
                    except ValueError:
                        pass
            clean_prompts.append(clean_text.strip(" ,"))
            batch_neg_data.append(neg_data)

        # 2. Safely preserve metadata (like .is_negative_prompt) for Forge compatibility
        def make_cond_obj(texts):
            if hasattr(prompt, "is_negative_prompt"):
                new_obj = type(prompt)(texts)
                for attr in ['is_negative_prompt', 'width', 'height']:
                    if hasattr(prompt, attr):
                        setattr(new_obj, attr, getattr(prompt, attr))
                return new_obj
            return texts

        # 3. Compile clean prompt (keeps the 'y' pooled vector pure for CFG 1.0)
        base_conds = model.orig_flux_forward(make_cond_obj(clean_prompts))
        
        if not has_neg:
            return base_conds

        # 4. Standardize the data structure to prevent Tensor.__contains__ crashes
        is_dict_return = isinstance(base_conds, dict)
        is_tensor_return = isinstance(base_conds, torch.Tensor)
        
        if is_dict_return:
            cond_list = [base_conds]
        elif is_tensor_return:
            cond_list = [{"txt": base_conds}]
        else:
            cond_list = [{"txt": c} if isinstance(c, torch.Tensor) else c for c in base_conds]

        _count = 0
        
        # 5. Process each batch item to append isolated negative embeddings safely
        for c_item in cond_list:
            if not isinstance(c_item, dict):
                continue
            
            txt_key = "crossattn" if "crossattn" in c_item else "txt"
            if txt_key not in c_item:
                continue

            base_txt = c_item[txt_key]
            
            if base_txt.ndim == 2:
                base_txt = base_txt.unsqueeze(0)
            
            b, base_seq_len, dim = base_txt.shape
            
            has_txt_ids = "txt_ids" in c_item
            if has_txt_ids:
                base_txt_ids = c_item["txt_ids"]
                if base_txt_ids.ndim == 2:
                    base_txt_ids = base_txt_ids.unsqueeze(0)

            out_txts = []
            out_masks = []
            out_txt_ids = []
            
            # Unpack the batch dimension to prevent token bleeding between prompts
            for idx in range(b):
                neg_list = batch_neg_data[idx] if idx < len(batch_neg_data) else []
                
                txt_list = [base_txt[idx]]
                mask_list = [torch.ones(base_seq_len, device=base_txt.device, dtype=base_txt.dtype)]
                
                if has_txt_ids:
                    txt_ids_list = [base_txt_ids[idx]]
                
                for (word, weight) in neg_list:
                    word_cond = model.orig_flux_forward(make_cond_obj([word]))
                    w_item = word_cond[0] if isinstance(word_cond, list) else word_cond
                    
                    if isinstance(w_item, torch.Tensor):
                        w_txt = w_item
                    else:
                        w_txt = w_item.get(txt_key, w_item.get("txt"))
                        
                    if w_txt is not None:
                        if w_txt.ndim == 3:
                            w_txt = w_txt[0]
                        elif w_txt.ndim == 1:
                            w_txt = w_txt.unsqueeze(0)
                            
                        w_seq_len = w_txt.shape[0]
                        txt_list.append(w_txt)
                        
                        w_mask = torch.full((w_seq_len,), weight, device=base_txt.device, dtype=base_txt.dtype)
                        mask_list.append(w_mask)
                        _count += 1
                        
                        if has_txt_ids:
                            w_txt_ids = w_item.get("txt_ids")
                            if w_txt_ids is not None:
                                if w_txt_ids.ndim == 3:
                                    w_txt_ids = w_txt_ids[0]
                                txt_ids_list.append(w_txt_ids)

                out_txts.append(torch.cat(txt_list, dim=0))
                out_masks.append(torch.cat(mask_list, dim=0))
                if has_txt_ids and len(txt_ids_list) == len(txt_list):
                    out_txt_ids.append(torch.cat(txt_ids_list, dim=0))

            # Pad all batch items sequentially to prevent shape mismatch in batched generations
            max_len = max([t.shape[0] for t in out_txts])
            padded_txts = []
            padded_masks = []
            padded_txt_ids = []

            for idx in range(b):
                pad_len = max_len - out_txts[idx].shape[0]
                if pad_len > 0:
                    padded_txts.append(F.pad(out_txts[idx], (0, 0, 0, pad_len), value=0.0))
                    padded_masks.append(F.pad(out_masks[idx], (0, pad_len), value=1.0))
                    if has_txt_ids and len(out_txt_ids) == b:
                        padded_txt_ids.append(F.pad(out_txt_ids[idx], (0, 0, 0, pad_len), value=0.0))
                else:
                    padded_txts.append(out_txts[idx])
                    padded_masks.append(out_masks[idx])
                    if has_txt_ids and len(out_txt_ids) == b:
                        padded_txt_ids.append(out_txt_ids[idx])

            c_item[txt_key] = torch.stack(padded_txts, dim=0)
            c_item["c_negpip_mask"] = torch.stack(padded_masks, dim=0).unsqueeze(-1)
            
            if has_txt_ids and len(padded_txt_ids) == b:
                c_item["txt_ids"] = torch.stack(padded_txt_ids, dim=0)

        if _count > 0:
            print(f"NegPiP Flux Enable (Isolated Targets: {_count})")

        if is_dict_return or is_tensor_return:
            return cond_list[0]
        else:
            return cond_list

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
