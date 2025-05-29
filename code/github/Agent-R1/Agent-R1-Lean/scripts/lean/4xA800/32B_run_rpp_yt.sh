ORM_HOME=/AI4M/users/ytwang

# Configure and validate env
source $ORM_HOME/init_env_yt_agent_r1.sh

# Experiment settings
verl_workdir=$ORM_HOME/auto-proof/Agent-R1-Lean
train_files=$verl_workdir/data/leanwkbk_agent_prompt_new/train.parquet
val_files=$verl_workdir/data/leanwkbk_agent_prompt_new/test.parquet
REF_MODEL_PATH=/AI4M/llm/Qwen2.5-32B-Instruct
PROJECT_NAME="Agent-R1-Lean"
EXPERIMENT_NAME="Qwen32B-klmse0.4"
TIMESTAMP=$(date +"%Y%m%d_%H")
LOG_DIR="$ORM_HOME/logs/ray_home/${TIMESTAMP}_node${NODE_RANK}_${PROJECT_NAME}_${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"
echo "Logging data in $LOG_DIR"

# Enter workdir
cd $verl_workdir

NUM_NODES=$PET_NNODES
export VLLM_ATTENTION_BACKEND=XFORMERS

export NODE_RANK=${PET_NODE_RANK:-0}
HOST_IP=$(hostname -i)

echo "Current node IP: $HOST_IP, NODE_RANK: $NODE_RANK"
echo "-- WORLD_SIZE: $WORLD_SIZE"
echo "-- NUM_NODES: $NUM_NODES"
echo "-- MASTER_ADDR: $MASTER_ADDR"
echo "-- MASTER_PORT: $MASTER_PORT"

if [ "$NODE_RANK" -eq 0 ]; then
  log_file="${LOG_DIR}/gpu_cpu_head_${NODE_RANK}.log"
else
  log_file="${LOG_DIR}/gpu_cpu_worker_${NODE_RANK}.log"
fi

## Ray head node initialization
if [ "$NODE_RANK" -eq 0 ]; then
  echo "[Head node] Init Ray head ..."
  # Write head node IP to shared file for worker nodes to read
  echo "$HOST_IP" > ${LOG_DIR}/.__ray_cluster_header
  
  # Start ray head node (adjust port and other parameters as needed)
  ray start --head --node-ip-address "$HOST_IP" --port=6379 \
      --dashboard-host "$HOST_IP" --dashboard-port=8265 \
      --ray-client-server-port=10001 --disable-usage-stats --log-color false --block &
  
  # Wait for worker node to start
  sleep 50
  
  #para should changed according to GPU number. Always make sure batch_size is divisible by GPU number.
  echo "[Head node] Start training task ..."
  PYTHONUNBUFFERED=1 python3 -u -m agent_r1.src.main_agent \
    data.train_files=./data/leanwkbk_agent_prompt_new/train.parquet \
    data.val_files=./data/leanwkbk_agent_prompt_new/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.max_start_length=4096 \
    data.max_tool_response_length=1024 \
    actor_rollout_ref.model.path=$REF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.4 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_repeat=8 \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=$REF_MODEL_PATH \
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
    tool.env='lean_search'
else
  echo "[Worker Node] Waiting for head node to start and join cluster ..."
  # Wait for enough time to ensure head node has written to shared file
  sleep 3
  header_ip=$(cat ${LOG_DIR}/.__ray_cluster_header)
  echo "[Worker Node] Join Ray cluster, Head IP: ${header_ip}:6379"
  ray start --address "${header_ip}:6379" --node-ip-address "$HOST_IP" --log-color false --block
fi

sleep inf