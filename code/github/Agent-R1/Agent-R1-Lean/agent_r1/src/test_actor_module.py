import ray
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoConfig
from verl.utils import hf_tokenizer
from verl.utils.model import update_model_config

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager



local_path = '/AI4M/users/qzh/lean_test/Agent/Temp/LeanRL/custom_model/Qwen2.5-7B-it'
override_model_config = {}
trust_remote_code = True
tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
torch_dtype = torch.float32
fsdp_size=-1

override_config_kwargs = {
    'bos_token_id': tokenizer.bos_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.pad_token_id,
}

override_config_kwargs.update(override_model_config)
update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
print(f'Model config after override: {actor_model_config}')

actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    config=actor_model_config,
                                                    attn_implementation='flash_attention_2',
                                                    trust_remote_code=trust_remote_code)
actor_module.cuda()

from verl.models.transformers.monkey_patch import apply_monkey_patch
apply_monkey_patch(model=actor_module)
actor_module.to(torch_dtype)


input_ids_rmpad = torch.load('input_ids_rmpad.pt').cuda()
position_ids_rmpad = torch.load('position_ids_rmpad.pt').cuda()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = actor_module(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=position_ids_rmpad,
        use_cache=False)
    print(output)
