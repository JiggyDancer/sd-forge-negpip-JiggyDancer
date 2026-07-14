from functools import wraps
import torch
import torch.nn.functional as F
from backend import attention 
from modules import shared

def patch_flux_negpip(instance, cls, *, unpatch=False):
    if len(cls._patched) < 3:
        cls._patched.append(False)
        
    if unpatch != cls._patched[2]:
        return

    cls._patched[2] = not cls._patched[2]

    model = shared.sd_model
    dit = model.forge_objects.unet.model.diffusion_model
    
    _hook_flux_dit_forward(instance, dit, unpatch)


def _hook_flux_dit_forward(instance, dit, remove: bool):
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
            tokens_data = getattr(shared.state, "negpip_flux_tokens", None)
            
            if tokens_data:
                b, s, d = v.shape
                txt_len = getattr(shared.state, "negpip_flux_txt_len", 0)
                is_cfg = getattr(shared.state, "negpip_flux_cfg", False)
                
                # CFG pass (cond and uncond batched)
                if is_cfg and b % 2 == 0:
                    v_c, v_uc = v.chunk(2)
                    t_c, t_uc = tokens_data
                    pad_c, pad_uc = getattr(shared.state, "negpip_flux_pad", (0, 0))
                    
                    if t_c > 0: 
                        end_c = txt_len - pad_c
                        v_c[:, end_c - t_c : end_c, :] = -v_c[:, end_c - t_c : end_c, :]
                    if t_uc > 0:
                        end_uc = txt_len - pad_uc
                        v_uc[:, end_uc - t_uc : end_uc, :] = -v_uc[:, end_uc - t_uc : end_uc, :]
                    
                    v = torch.cat([v_c, v_uc], dim=0)
                # CFG 1.0 pass (single batched condition)
                else:
                    t_c = tokens_data if isinstance(tokens_data, int) else tokens_data[0]
                    if t_c > 0:
                        v[:, txt_len - t_c : txt_len, :] = -v[:, txt_len - t_c : txt_len, :]
                            
            return attention.orig_flux_negpip_attention(q, k, v, heads, mask, *args, **kwargs)

        attention.attention_function = negpip_flux_attention

    @torch.inference_mode()
    @wraps(dit.orig_flux_forward)
    def negpip_forward(*args, **kwargs):
        args_list = list(args)
        
        txt = kwargs.get("txt", args_list[2] if len(args_list) > 2 else None)
        txt_ids = kwargs.get("txt_ids", args_list[3] if len(args_list) > 3 else None)
        
        if instance and getattr(instance, "active", False) and txt is not None:
            c_add = instance.conds[0] if getattr(instance, "conds", None) else None
            uc_add = instance.unconds[0] if getattr(instance, "unconds", None) else None
            t_c = instance.c_tokens[0] if c_add is not None else 0
            t_uc = instance.uc_tokens[0] if uc_add is not None else 0
            
            is_cfg = txt.shape[0] == instance.batch_size * 2
            
            if is_cfg:
                txt_c, txt_uc = txt.chunk(2)
                if c_add is not None: txt_c = torch.cat([txt_c, c_add.to(txt.device, dtype=txt.dtype)], dim=1)
                if uc_add is not None: txt_uc = torch.cat([txt_uc, uc_add.to(txt.device, dtype=txt.dtype)], dim=1)
                
                max_len = max(txt_c.shape[1], txt_uc.shape[1])
                pad_c = max_len - txt_c.shape[1]
                pad_uc = max_len - txt_uc.shape[1]
                
                if pad_c > 0: txt_c = F.pad(txt_c, (0, 0, 0, pad_c), value=0.0)
                if pad_uc > 0: txt_uc = F.pad(txt_uc, (0, 0, 0, pad_uc), value=0.0)
                
                new_txt = torch.cat([txt_c, txt_uc], dim=0)
                
                if txt_ids is not None:
                    txt_ids_c, txt_ids_uc = txt_ids.chunk(2)
                    if c_add is not None: 
                        pad_ids = torch.zeros((txt_ids_c.shape[0], c_add.shape[1], txt_ids_c.shape[2]), device=txt_ids.device, dtype=txt_ids.dtype)
                        txt_ids_c = torch.cat([txt_ids_c, pad_ids], dim=1)
                    if uc_add is not None:
                        pad_ids = torch.zeros((txt_ids_uc.shape[0], uc_add.shape[1], txt_ids_uc.shape[2]), device=txt_ids.device, dtype=txt_ids.dtype)
                        txt_ids_uc = torch.cat([txt_ids_uc, pad_ids], dim=1)
                        
                    if pad_c > 0: txt_ids_c = F.pad(txt_ids_c, (0, 0, 0, pad_c), value=0.0)
                    if pad_uc > 0: txt_ids_uc = F.pad(txt_ids_uc, (0, 0, 0, pad_uc), value=0.0)
                    new_txt_ids = torch.cat([txt_ids_c, txt_ids_uc], dim=0)
                else:
                    new_txt_ids = None
                    
                shared.state.negpip_flux_cfg = True
                shared.state.negpip_flux_tokens = (t_c, t_uc)
                shared.state.negpip_flux_pad = (pad_c, pad_uc)
                shared.state.negpip_flux_txt_len = new_txt.shape[1]
                
            else:
                new_txt = txt
                new_txt_ids = txt_ids
                if c_add is not None:
                    new_txt = torch.cat([txt, c_add.to(txt.device, dtype=txt.dtype)], dim=1)
                    if txt_ids is not None:
                        pad_ids = torch.zeros((txt_ids.shape[0], c_add.shape[1], txt_ids.shape[2]), device=txt_ids.device, dtype=txt_ids.dtype)
                        new_txt_ids = torch.cat([txt_ids, pad_ids], dim=1)
                        
                shared.state.negpip_flux_cfg = False
                shared.state.negpip_flux_tokens = t_c
                shared.state.negpip_flux_pad = (0, 0)
                shared.state.negpip_flux_txt_len = new_txt.shape[1]
                
            if "txt" in kwargs: kwargs["txt"] = new_txt
            elif len(args_list) > 2: args_list[2] = new_txt
            
            if new_txt_ids is not None:
                if "txt_ids" in kwargs: kwargs["txt_ids"] = new_txt_ids
                elif len(args_list) > 3: args_list[3] = new_txt_ids

        try:
            res = dit.orig_flux_forward(*args_list, **kwargs)
        finally:
            shared.state.negpip_flux_tokens = None
            
        return res

    negpip_forward._negpip = True
    dit.forward = negpip_forward
