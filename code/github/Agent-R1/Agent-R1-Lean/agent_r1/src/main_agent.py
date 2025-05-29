# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from .agent_ray_trainer import RayAgentTrainer

from agent_r1.tool import ToolEnv
from agent_r1.tool.tools import _default_tools
from agent_r1.src.utils.model_output_logger import ModelOutputLogger

import os
import ray
import hydra

from verl import DataProto
from .reward_score import _default_compute_score_format, _default_compute_score_answer, _default_compute_score_format_answer
import torch

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, log_outputs=False, is_validation=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.log_outputs = log_outputs
        self.is_validation = is_validation
        
        # 初始化输出日志记录器
        if self.log_outputs:
            self.logger = ModelOutputLogger()

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        answer_lst = []
        format_lst = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            valid_response_ids = response_ids[:valid_response_length].long()

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]
            if not sequences_str.endswith(self.tokenizer.eos_token):
                sequences_str += self.tokenizer.eos_token

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']

            score = _default_compute_score_format_answer(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            answer_score = _default_compute_score_answer(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            format_score = _default_compute_score_format(data_source=data_source, solution_str=sequences_str)

            answer_lst.append(answer_score)
            format_lst.append(format_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt+response]", sequences_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                
            # 记录所有样本的输出到日志文件
            if self.log_outputs:
                # 提取提示部分和响应部分
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                
                # 构建元数据
                metadata = {
                    "data_source": data_source,
                    "format_score": format_score,
                    "answer_score": answer_score
                }
                
                # 根据是验证还是训练选择不同的日志方法
                if self.is_validation:
                    self.logger.log_val_sample(
                        prompt=prompt_str,
                        response=response_str, 
                        ground_truth=ground_truth,
                        score=score,
                        metadata=metadata
                    )
                else:
                    self.logger.log_train_sample(
                        prompt=prompt_str,
                        response=response_str, 
                        ground_truth=ground_truth,
                        score=score,
                        metadata=metadata
                    )

        return reward_tensor, answer_lst, format_lst

@hydra.main(config_path='config', config_name='agent_trainer', version_base=None)
def main(config):
    run_agent(config)


def run_agent(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from .fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from .agent_ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        tools = _default_tools(config.tool.env)
        env = ToolEnv(tools=tools, max_turns=config.tool.max_turns)
        
        # Set default values when not configured
        num_examine_train = 0
        if hasattr(config.data, 'num_examine_train') and config.data.num_examine_train:
            num_examine_train = config.data.num_examine_train
            
        num_examine_val = 1
        if hasattr(config.data, 'num_examine_val') and config.data.num_examine_val:
            num_examine_val = config.data.num_examine_val
            
        # 创建训练和验证的奖励管理器，启用日志记录
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=num_examine_train, log_outputs=True, is_validation=False)
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=num_examine_val, log_outputs=True, is_validation=True)

        trainer = RayAgentTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn,
                            env=env)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
