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
                    w_item = word_cond
