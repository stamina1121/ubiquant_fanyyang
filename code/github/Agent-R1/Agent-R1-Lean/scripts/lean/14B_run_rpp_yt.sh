#!/bin/bash

# 设置环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
# export BASE_MODEL='/AI4M/users/qzh/lean_test/Agent/Temp/LeanRL/custom_model/Qwen2.5-7B-it'
export BASE_MODEL='/AI4M/llm/Qwen2.5-14B-Instruct'
export PROJECT_NAME='Agent-R1-Lean'
export EXPERIMENT_NAME="Qwen14BIT_kllowvar_pr_$(date +%Y%m%d_%H%M%S)"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# 创建日志目录
LOG_DIR="log/${PROJECT_NAME}"
mkdir -p "${LOG_DIR}"

# 设置日志文件路径
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

# 记录启动信息
{
    echo "===== 任务启动信息 ====="
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "实验名称: ${EXPERIMENT_NAME}"
    echo "项目名称: ${PROJECT_NAME}"
    echo "基础模型: ${BASE_MODEL}"
    echo "当前目录: $(pwd)"
    echo "执行命令: $0 $@"
    echo "任务PID: $$"
    echo "父PID: $PPID"
    echo "用户: $(whoami)"
    echo "主机: $(hostname)"
    
    echo -e "\n===== 环境变量 ====="
    printenv | grep -E 'VLLM_|BASE_MODEL|PROJECT_NAME|EXPERIMENT_NAME|HYDRA|CUDA'
    echo -e "\n===== 执行的脚本内容 ====="
    cat "$0"
    echo -e "\n===== 开始执行 ====="
} > "${LOG_FILE}" 2>&1

# 记录后台任务PID
echo "主程序PID: $!" >> "${LOG_FILE}"
echo "任务已启动，日志文件: ${LOG_FILE}"
echo "使用以下命令跟踪日志:"
echo "tail -f ${LOG_FILE}"
#     data.train_files=./data/hotpotqa/train.parquet \
#     data.test_files=./data/hotpotqa/test.parquet \
export RAY_DEBUG_POST_MORTEM=1
nohup python3 -u -m agent_r1.src.main_agent \
    data.train_files=./data/leanwkbk_agent_prompt_new/train.parquet \
    data.val_files=./data/leanwkbk_agent_prompt_new/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.max_start_length=8192 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_repeat=2 \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty=kl \
    algorithm.use_process_rewards=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    ++reward_model.reward_manager=prime \
    tool.max_turns=3 \
    tool.env='lean_search' $@ >> "${LOG_FILE}" 2>&1

