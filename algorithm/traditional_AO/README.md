# Traditional AO reference

此目录独立实现传统交替优化，只读取 `system_model` 和 MAPPO 的环境构建、绘图及文件保存函数，不修改它们。支持工程当前的单 Bob、分组有源/无源 RIS 模型。

## 算法和信息条件

AO 不需要 Actor/Critic 训练。每个时隙在当前信道上求解一次局部优化，再调用真实环境的 `step`。episode 曲线是跨随机布局的仿真表现，不应解释成神经网络学习曲线。

优化变量与 MAPPO 的可执行动作保持一致：

- 归一化复卫星波束，投影到单位功率球，环境再换算为实际发射功率。
- 每组一个相位残差，范围为参数文件中的 `ris_phase_residual_max_deg`；逐单元最终相位为 Bob 解析基相位加对应组残差。更新波束时也重新计算解析基相位。
- 每组一个有源/无源二值模式，再通过 `element_to_group` 展开到全部单元。
- 有源幅度、RIS RF 输出预算、噪声、干扰方向响应和不可行时的无源修复全部使用原环境的实现。

每时隙以 MRT + 零残差的全有源、全无源方案，以及前一时隙解中较优者初始化。依次执行：波束有限差分梯度与投影回溯、组相位有限差分梯度与投影回溯、各组模式的贪心翻转。仅接受超过容差的改善，达到迭代上限或整轮改善不足时结束。它是投影 AO/块坐标局部优化，没有使用 SCA/SDR，不保证全局最优或全局性能上界。

目标是最大化当前时隙的最小保密裕量 `min(C_B - max(C_E) - R0)`。SOP、环境奖励和容量仍按原环境计算。该瞬时目标与整段轨迹 SOP 最小化有关，但二者不等价，也不保证 AO 总是比 MAPPO 好。

**信息标记为 `oracle_full_csi`。** AO 使用真实 Eve 信道及当前 G2G 衰落进行候选评估；MAPPO Actor 通常没有这些信息。因此它适合作为理想 CSI 参考算法，论文应明确披露信息优势，不能声称是与局部观测 MAPPO 完全相同信息下的比较，也不能称为严格上界。

原环境有部分随机信道在 `_evaluate` 内抽取。求解器只操作环境的深拷贝，每个候选回退到相同的信道 RNG 状态和前一时隙指标，保留固定 CSI、原干扰方向响应和原功率投影。候选搜索不消耗真实环境的随机序列。最后真实 `env.step` 使用该解；测试验证预测裕量、奖励与实际执行一致。若将来原环境改变了内部状态更新方式，应重新运行这些一致性测试。

## 运行

在工程根目录先进行短实验：

```powershell
python -B -m algorithm.traditional_AO.main --episodes 2 --max_episode_steps 5 --ao-max-iterations 1
```

与此前 MAPPO 的 8000 episode / 160 万训练交互对齐的仿真预算：

```powershell
python -B -m algorithm.traditional_AO.main --episodes-per-update 16 --total-updates 500 --max_episode_steps 200 --seed 2026 --jammer_power 0.1 --g2g_reference_gain_db -58.5 --num_ris_elements 256 --num_ris_groups 64
```

这里 update 只表示汇总、绘图和进度保存的边界，不是一次策略梯度更新。`--episodes` 与 `--total-updates` 二选一。其余系统参数均传给现有 `get_system_params`，未提供时跟随当前参数文件。也支持直接运行 `python -B algorithm/traditional_AO/main.py`。

AO 每个时隙需要多次模型计算，完整仿真可能明显慢于 RL 策略推理。有限差分每轮的连续梯度约需 `2*(2*M+G)` 次评估，再加回溯和 G 次模式翻转。先用短实验的 `mean_solve_seconds` 估算整体耗时；不能将候选模型调用当作额外环境时隙，也应将其计算成本计入传统算法开销。

可调参数：`--ao-max-iterations` 默认 3，`--ao-gradient-step` 默认 0.25，`--ao-finite-difference` 默认 0.001。求解器使用 NumPy；绘图复用 MAPPO，因此保留工程原有 Matplotlib/PyTorch 依赖。无需额外安装 CVX 或商业求解器。创建本模块时只运行短验证，不自动启动完整仿真。

## 输出

每次输出到独立的 `algorithm/traditional_AO/result/时间戳/`：

- `episode_metrics.csv`：列名、顺序和统计口径与 MAPPO 一致；`policy_version=0` 表示无学习策略版本。
- `update_metrics.csv`：保留 MAPPO 列顺序，末尾增加 AO 的初始/最终裕量、迭代次数、模型评估次数和耗时。Actor/Critic loss、熵、KL 等不适用列留空，不用零值伪造。
- `training_curves.png`：与 MAPPO 相同的 3×2 布局和窗口 20。四个性能面板沿用 SOP、return、Bob 容量、最大 Eve 容量；两个顶部面板显示 AO 前后裕量和单时隙求解耗时。
- `ao_iterations.csv`：每时隙的逐轮目标与三个变量块的改善量，可检验同一固定信道下的单调性。
- `step_metrics.csv`：逐时隙容量、中断、奖励、功率、请求模式与实际模式、求解耗时。
- `config.json`、`summary.json`、`checkpoint.json`：系统参数、AO 配置、汇总与进度。`checkpoint.json` 是进度记录，不是可恢复训练的网络检查点；本版没有续跑入口。
- `training_error.log`：运行异常时的堆栈。

保密差、容量均以 bit/s/Hz 表示。当前单 Bob 下 `mean_bob_capacity` 与逐时隙最大 Bob 容量的回合均值相同；Eve 是每时隙最大 Eve 容量的回合均值，SOP 是中断时隙比例。不同 episode 重新采样布局，容量曲线无需随 episode 单调改善；AO 内部的单调性应查看同一时隙的 `ao_iterations.csv`。

## 核实

```powershell
python -B -m unittest discover -s algorithm/traditional_AO/tests -v
```

测试覆盖组动作、功率与相位边界、候选随机数一致性、真实环境不受搜索影响、目标单调、实际奖励重放、强功率约束的原模型修复，以及 MAPPO 列结构和输出统计。验证不能代替正式多种子性能实验。
