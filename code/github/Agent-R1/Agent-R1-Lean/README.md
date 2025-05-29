# Agent-R1 in Lean

## Environment Setup
标准环境可按以下脚本生成

```bash
conda create -n verl python==3.10
conda activate verl
# install verl together with some lightweight dependencies in setup.py
pip3 install torch==2.4.0
pip3 install flash-attn --no-build-isolation
git submodule init
git submodule update
cd verl
pip3 install -e .
pip3 install faiss-cpu FlagEmbedding
```

## Train in Lean
默认脚本用 Qwen2.5-7B-Instruct 训练，Qwen2.5-14B-Instruct 的相关脚本在 `scripts/lean/14B_run_rpp_yt.sh`
```bash
# 生成数据集,可以换其他数据集,默认ds_prover
bash scripts/lean/create_lean_dataset.sh
# 跑训练脚本
bash scripts/lean/run_rpp.sh
# 防止内存泄露
nohup python scripts/kill_all.py
```

## Important Scripts
1. 工具列表
   1. `agent_r1/tool/tools/leansearch_tool.py` 搜索工具
   2. `agent_r1/tool/tools/leanverify_tool.py` repl 工具
2. 工具说明
   1. 核心函数为batch_execute; 工具定义完成后在agent_r1/tool/tools/__init__.py处进行注册环境与对应的工具
   2. 于训练脚本的tool.env处指定环境名称,根据注册的环境名对应的工具分配工具
3. 奖励函数
   1. `agent_r1/src/reward_score/lean.py` 处定义奖励函数
   2. 目前奖励函数仍为 strict 模式，正在更新中
   3. **在奖励函数处设置 lean 检验相关配置**
4. 数据生成
   1. `examples/data_preprocess/lean.py` 处生成数据集
   2. 用 `scripts/temp_transform.py` 可以临时把原数据集 prompt 更换为 agent_prompt，这个文件里可以浏览现在的 prompt.

## Known Issues

1. 必须有 ppo_mini_batch_size >= n_gpus_per_node * ppo_micro_batch_size 否则第一步 NaN
2. use_kl_loss=True, kl_loss_type=low_var_kl 且 kl_loss_coef 较小时，可能出现梯度爆炸问题，目前可以设置 use_kl_loss=false 或 kl_loss_coef=0.4 或 kl_loss_type=mse 来缓解

## TODO
1. 完整测试更换为 4.16.0 检验后，训练的稳定性
2. 把检验环境和工具调用环境替换为 repl_server，启用 relaxed mode
3. 基于日期的训练主 log 整理（目前 log 太复杂）
4. lean_verify 工具的测试（如何高效返回工具结果，最多允许模型调用多少次 verify，模型调用 verify 工具时应采用什么格式，如何给予调用工具的 process format reward）
5. 把此前所有数据集制备为 agent-r1 支持的形式，并对代数数据进行测试
6. 用 claude agent 蒸馏热启动数据，提高模型训练稳定性


