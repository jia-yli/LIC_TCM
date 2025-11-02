import os
import re
import torch
from tqdm import tqdm
from models import TCM, TCMWeighted

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def main(tcm_ckpt_path, tcm_weighted_ckpt_path):
  assert tcm_ckpt_path != tcm_weighted_ckpt_path
  all_tcm_ckpts = sorted([f for f in os.listdir(tcm_ckpt_path) if f.endswith('.pth.tar')])
  for tcm_ckpt_file in tqdm(all_tcm_ckpts):
    match_n = re.search(r'[nN]_(\d+)', tcm_ckpt_file)
    N = int(match_n.group(1))
    net_tcm = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N, M=320)
    net_tcm_weighted = TCMWeighted(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N, M=320)
    
    # load tcm
    tcm_checkpoint = torch.load(os.path.join(tcm_ckpt_path, tcm_ckpt_file))
    # import pdb;pdb.set_trace()
    dictory = {}
    for k, v in tcm_checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    net_tcm.load_state_dict(dictory)

    # load state for tcm_weighted
    update_registered_buffers(
        net_tcm_weighted.gaussian_conditional,
        "gaussian_conditional",
        ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        dictory,
    )

    # load state
    for name, submodule in net_tcm_weighted.named_children():
      # other modules
      if name not in ["g_a", "g_s", "entropy_bottleneck_b"]:
        tcm_submodule = getattr(net_tcm, name)
        submodule.load_state_dict(tcm_submodule.state_dict(), strict=True)

    net_tcm_weighted.g_a.layer1.load_state_dict(net_tcm.g_a[0].state_dict(), strict=True)
    for idx in range(1, 4):
      tcm_submodules = eval(f"net_tcm.m_down{idx}")
      num_submodules = len(tcm_submodules)
      assert len(eval(f"net_tcm_weighted.g_a.m{idx}")) == len(tcm_submodules) - 1
      for submodule_idx in range(num_submodules-1):
        eval(f"net_tcm_weighted.g_a.m{idx}")[submodule_idx].load_state_dict(tcm_submodules[submodule_idx].state_dict(), strict=True)
      eval(f"net_tcm_weighted.g_a.down{idx}").load_state_dict(tcm_submodules[num_submodules-1].state_dict(), strict=True)

    net_tcm_weighted.g_s.layer1.load_state_dict(net_tcm.g_s[0].state_dict(), strict=True)
    for idx in range(1, 4):
      tcm_submodules = eval(f"net_tcm.m_up{idx}")
      num_submodules = len(tcm_submodules)
      assert len(eval(f"net_tcm_weighted.g_s.m{idx}")) == len(tcm_submodules) - 1
      for submodule_idx in range(num_submodules-1):
        eval(f"net_tcm_weighted.g_s.m{idx}")[submodule_idx].load_state_dict(tcm_submodules[submodule_idx].state_dict(), strict=True)
      eval(f"net_tcm_weighted.g_s.up{idx}").load_state_dict(tcm_submodules[num_submodules-1].state_dict(), strict=True)

    os.makedirs(tcm_weighted_ckpt_path, exist_ok=True)
    torch.save(
      {"state_dict": net_tcm_weighted.state_dict()},
      os.path.join(tcm_weighted_ckpt_path, f"converted_tcm_weighted_{tcm_ckpt_file}")
    )

if __name__ == "__main__":
  tcm_ckpt_path = "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained"
  tcm_weighted_ckpt_path = "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained_tcm_weighted"
  main(tcm_ckpt_path, tcm_weighted_ckpt_path)