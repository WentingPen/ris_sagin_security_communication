# Satellite–RIS IPPO baseline

本目录新增 IPPO，不修改 system_model 或 algorithm/ppo，也不会启动、暂停或控制已有的 MAPPO 训练进程。

## 设计与公平对比

- 直接复用 MAPPO 的 SatelliteActor、HAPSRISActor、RiskGNN 和环境构建函数，不复制、修改物理模型。
- 保留合法链路瞬时 CSI、统计风险图、当前卫星波束条件输入、逐单元解析基相位、组模式和组相位残差，以及环境内的功率投影。
- 两个 LocalCritic 各有独立的 GNN、MLP、优化器、价值估计、GAE、回报目标；每个 Actor 只使用自己 Critic 的优势。
- 两个 Critic 均只读取公开的 observation["vector"] 和 observation["risk_nodes"]。当前模型给两个智能体相同的公开观测，所以 Critic 输入内容相同，但参数和更新彼此独立。
- RIS Actor 保留与 MAPPO 一致的波束消息；两个 Critic 都在联合动作生成前的公开观测上估值，不接收其他智能体动作或环境的私有 critic_observation。
- 全局奖励保持不变。Eve 容量等 info 字段只用于输出统计，不作为 Critic 或 Actor 输入。
- 默认学习率、PPO clipping、GAE、熵退火、KL 提前停止、梯度裁剪、每批回合数与 MAPPO 一致。

读取现有 system_model/params.py，不把 RIS 单元数、分组、信道或 Eve 分布硬编码到 IPPO。新训练跟随当前参数文件，恢复训练使用 checkpoint 中的参数。

## 启动

在工程根目录运行：

~~~powershell
python -m algorithm.IPPO.main
~~~

总更新次数和系统参数的覆盖方式与 MAPPO 相同：

~~~powershell
python -m algorithm.IPPO.main --total-updates 1000 --seed 42
~~~

默认使用可用的 CUDA。需要独立 CPU 进程时：

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -B -m algorithm.IPPO.main --device cpu --total-updates 1000
~~~

也支持直接运行 python algorithm/IPPO/main.py。创建本目录时只做短时 CPU 验证，不自动启动完整 IPPO 训练。

## 输出

每次写入新的 algorithm/IPPO/result/时间戳/，不写入 MAPPO 的结果目录：

- episode_metrics.csv：与 MAPPO 完全相同的列和指标口径。
- update_metrics.csv：保留 MAPPO 原有列，末尾增加 satellite_critic_loss、haps_ris_critic_loss；critic_loss 为两者算术平均。
- training_curves.png：同样的 3×2 图，包含 Actor loss、Critic loss、SOP、episode return、Bob 容量、最大 Eve 容量；Critic 面板显示两个局部 Critic 和平均值。
- checkpoint.pt：每个完整 update 后原子保存四个网络、四个优化器、配置和计数器。
- training_error.log：训练异常时保存堆栈。

容量曲线与 MAPPO 一样，是每个 episode 内“逐时隙最大容量”的平均值；当前只有一个 Bob。SOP 为该 episode 的中断时隙占比。滑动平均窗口同样为 20 个 episode。

## 恢复和 phase-only

~~~powershell
python -m algorithm.IPPO.main --resume "algorithm/IPPO/result/某次运行/checkpoint.pt" --total-updates 2000
python -m algorithm.IPPO.main --phase-only-training --total-updates 1000
~~~

恢复会创建新结果目录，沿用保存的系统参数与 IPPO 配置，并从已完成 update 的计数继续；新 CSV 只记录恢复后的数据。与现有 MAPPO 一样，恢复不保证随机信道和动作序列逐位复现。

checkpoint 标记为 algorithm="IPPO"，包含两个局部 Critic。MAPPO checkpoint 不能作为 IPPO checkpoint 恢复。phase-only 模式保留 MRT、全无源 RIS、固定回合种子和 gamma=gae_lambda=0，卫星 Actor 与模式头冻结。

## 核实

~~~powershell
python -B -m unittest discover -s algorithm/IPPO/tests -v
~~~

测试仅使用 CPU、单线程和本目录下的临时输出；检查独立 Critic/优势、私有信息隔离、组动作、梯度更新、截断时的 GAE bootstrap、checkpoint、输出兼容和 phase-only 模式。完整训练收敛表现需要正式实验验证。
