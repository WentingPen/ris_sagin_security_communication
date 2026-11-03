# 当前 SAGIN–RIS 工程的保密差下界与 SOP 理论界

日期：2026-11-01。本文是与当前工程一致的分析推导，不修改环境、策略网络或训练参数。以下“容量”沿用代码术语，指采用 Gaussian signaling、把干扰当噪声时的频谱效率；Eve 非串谋，以最大单个 Eve 容量为威胁指标。

## 1. 已实现内容与推导范围

- `system_model/system_env.py:503-513` 已实现 SINR→容量→最大 Eve 容量→保密差→中断事件。
- `system_model/system_env.py:821` 给回合中断频率，并非解析 SOP 分布。
- `../2026-11-01_sop64/bound_derivation.md` 已有任意动作的保密差乐观**上界**，用于不可达性核查；它不能称为保密差下界。
- 本文新增推导：当前分组残差结构的保密差下界、SOP 界、当前 HCPP 模型的空间概率表达，以及纯 PPP 可用的条件。

主推导条件：一个 Bob、一个数据流；按当前配置干扰发射功率 P_J=0；所有振幅均指 RF 功率投影后的真实执行振幅。下界亦可在非零干扰时逐点应用，但必须保留实际 Bob 干扰或使用其可靠上界，不能沿用零干扰的统计简化。

## 2. 信道、噪声与目标

令 M 为卫星天线数，N 为 RIS 单元数，G 为分组数。定义：

- H∈C^(N×M)：卫星→RIS 信道；h_k∈C^(1×M)：卫星→接收端 k 直达信道；
- g_k∈C^N：RIS→k 信道；w∈C^M：执行波束，||w||²≤P；v=Hw；
- m_n∈{0,1}：单元有源模式；ρ_n：执行振幅；φ_n=ρ_n exp(jθ_n)。无源振幅为1。

与代码的行向量乘法约定一致，有效信道为

$$
h_k^{\rm eff}=h_k+(g_k\odot\phi)^T H,
\qquad s_k=h_kw+\sum_n g_{kn}\phi_n v_n.
$$

单流下接收端总噪声、干扰功率为

$$
Q_k=\sigma_k^2+\sigma_R^2\sum_n m_n\rho_n^2|g_{kn}|^2+J_k,
\qquad \gamma_k=\frac{|s_k|^2}{Q_k}.
$$

J_B 是 active Eve 经过直达及 RIS 路径的干扰；被动 Eve 的 J_e 含干扰抵消后的残余；active Eve 的 J_e 含直接残余自干扰与 RIS 回波。当前 P_J=0，三者均为零，但**有源 RIS 噪声仍然存在**。对应 `legitusers.py:127-145`、`hcpp_eve.py:180-196,312-326`。

执行振幅须满足

$$
P_{\rm RIS}=\sum_n m_n\rho_n^2
\left(|v_n|^2+P_J|f_n|^2+\sigma_R^2\right)
\le P_{\rm RIS}^{\max}.
$$

不能不经检查就把所有有源振幅固定为8。当前 `uav_ris.py:284-324` 对选中的有源单元实施共同增益投影；模式不可行时可回退无源。本文使用实际执行值，因此保留该约束。

定义

$$
\gamma_E=\max_{e\in\Phi}\gamma_e,\quad
C_B=\log_2(1+\gamma_B),\quad
C_E^{\max}=\log_2(1+\gamma_E),
$$
$$
D=C_B-C_E^{\max},\quad C_s=[D]^+,\quad M_s=D-R_0.
$$

代码 `raw_security_gap`/训练日志 `gap` 是 D，不是 M_s。由于当前 R_0=0.1>0，严格地有

$$
\{C_s<R_0\}=\{D<R_0\}.
$$

令 q=2^(R_0)，则

$$
\boxed{\mathrm{SOP}=\Pr\{\gamma_B<q(1+\gamma_E)-1\}.}
$$

Bob 与 Eve 共享 H、波束、RIS 动作，通常不能假设两者 SINR 独立。连续分布情形的通用表达为

$$
\mathrm{SOP}=\int_0^\infty
F_{\gamma_B|\gamma_E}\big(q(1+y)-1\mid y\big)
\,dF_{\gamma_E}(y).
$$

若分布含原子，严格事件应使用左极限 CDF。对本工程的连续衰落，门限等号通常为零概率事件。回合中训练曲线计算的是经验频率

$$
\widehat p_{\rm out}=T^{-1}\sum_{t=1}^T\mathbf1\{D_t<R_0\},
$$

而不是上述概率的闭式值。同回合几何固定，不能把所有时隙自动视为无条件独立试验。

## 3. 解析基相位使 Bob 有可证明的下界

工程 `system_env.py:273-296` 依据执行波束给出

$$
\theta_{0n}=\arg(h_Bw)-\arg(g_{Bn}v_n).
$$

Actor 输出分组模式、分组残差；单元实际相位为

$$
\theta_n=\theta_{0n}+\delta_{g(n)},\qquad
|\delta_g|\le\Delta.
$$

当前 MAPPO 的 Δ=30°（`algorithm/ppo/Controller.py:35`、`main.py:468`），不是 `params.py` 中另一处遗留默认45°。以下要求 0≤Δ≤π/2。

令

$$
d=|h_Bw|,\quad A_g=\sum_{n\in g}\rho_n|g_{Bn}v_n|,
\quad A_B=\sum_g A_g.
$$

消去共同相位后：

$$
s_B=e^{j\arg(h_Bw)}\left(d+\sum_gA_ge^{j\delta_g}\right).
$$

取实部并利用 |z|≥Re(z)，得到

$$
|s_B|\ge d+\sum_gA_g\cos\delta_g
\ge d+\cos\Delta\,A_B.
$$

所以

$$
\boxed{\underline\gamma_B=
\frac{(d+\cos\Delta\,A_B)^2}{Q_B}\le\gamma_B.}
$$

零残差时 |s_B|=d+A_B，Bob 幅度精确相干叠加。Δ=30° 时 cosΔ≈0.8660，cos²Δ=0.75；固定波束和执行振幅，实际总信号功率不低于完全相干值的75%，相应最大损失约1.25 dB。直连幅度 d>0 时一般更小。

这说明当前工程的主要相干叠加由解析基相位完成，并不是全由 RL 学习。界使用的是解析基相位加组残差，不能套给“组内最终相位完全相同”的另一种模型。代码对小于10^(-30)的反射乘积跳过基相位；极端机器级严谨界可扣除这些项最大总幅度，正常随机样本没有实际影响。

## 4. Eve 容量上界

首先，三角不等式直接给出

$$
|s_e|\le |h_ew|+\sum_n\rho_n|g_{en}v_n|.
$$

它允许 Eve 各反射项全部相干，通常偏保守。可保留组内相消得到更紧的界。令

$$
a_e=h_ew,\quad
k_{eg}=\sum_{n\in g}\rho_ng_{en}v_ne^{j\theta_{0n}},\quad
K_e=\sum_g|k_{eg}|.
$$

Eve 信号为 a_e+Σ_g k_eg exp(jδ_g)，因而有两个上界：

$$
|s_e|\le |a_e|+K_e,
$$
$$
|s_e|\le\left|a_e+\sum_gk_{eg}\right|
+\sum_g|k_{eg}|\,|e^{j\delta_g}-1|
\le\left|a_e+\sum_gk_{eg}\right|+2\sin(\Delta/2)K_e.
$$

取两者较小者

$$
U_e=\min\left\{|a_e|+K_e,
\left|a_e+\sum_gk_{eg}\right|+2\sin(\Delta/2)K_e\right\}.
$$

保留有源 RIS 噪声，丢弃 Eve 侧非负额外干扰，得到分母下界

$$
Q_{e0}=\sigma_E^2+\sigma_R^2\sum_nm_n\rho_n^2|g_{en}|^2\le Q_e.
$$

于是

$$
\boxed{\gamma_e\le\overline\gamma_e=U_e^2/Q_{e0},\qquad
\gamma_E\le\max_e\overline\gamma_e.}
$$

当前 P_J=0 时 Q_e=Q_e0，因此未牺牲额外的分母精度。若需进一步简化，也可用 σ_E² 作为分母，但界会更松。每个 Eve 分别取上界再取最大，未要求它们最坏相位能够同时实现，因此保守方向正确。

## 5. 保密差下界与 SOP 两侧界

log2(1+x) 单调递增，因此

$$
\boxed{D\ge D_L=
\log_2\frac{1+\underline\gamma_B}
{1+\max_e\overline\gamma_e}.}
$$
$$
\boxed{C_s\ge[D_L]^+.}
$$

D_L 允许为负；不能为了好看把它先截断成0再称为原始 D 的下界，因为 D 本身可能为负。截断只适用于 C_s。

由 D≥D_L 得事件包含关系 {D<R_0}⊆{D_L<R_0}，所以

$$
\boxed{\mathrm{SOP}\le\Pr(D_L<R_0).}
$$

已有诊断文档中的任意动作上界 D≤D_U 则给出相反方向：

$$
\boxed{\Pr(D_U<R_0)\le\mathrm{SOP}\le\Pr(D_L<R_0).}
$$

同样的逐样本事件包含关系也约束对应经验频率，但经验频率不是总体概率的置信界。D_L 使用实际 Eve CSI，可在仿真器中离线验证，不能直接视为当前 Actor 可观测的输入。

有衰落时，论文更有用的目标是概率保证。若

$$
\Pr(\gamma_B\ge b)\ge1-\epsilon_B,\quad
\Pr(\gamma_E\le e)\ge1-\epsilon_E,
$$

则无需独立性，并集界给出

$$
\Pr\!\left(D\ge\log_2\frac{1+b}{1+e}\right)
\ge1-\epsilon_B-\epsilon_E.
$$

当 log2[(1+b)/(1+e)]≥R_0，就有 SOP≤ε_B+ε_E。这里 b 是 Bob SINR 下分位数、e 是最大 Eve SINR 上分位数，不能用平均容量代替。

优化值也要区别：若 D*(ω)=sup_{可行动作}D(a,ω)，任意可行的 MRT＋零残差相干参考策略满足 D*≥D_ref≥D_L,ref。这不证明训练得到的 MAPPO 已达到全局最优，也不证明它必然胜过参考策略。

## 6. 当前空间模型是条件非空 HCPP

代码 `hcpp_eve.py:45-104`：先采样非齐次父 PPP，删除 Bob 保护区内的点，再按独立随机标记进行 Matérn-II 硬核稀疏化，最后拒绝空点集。主动 Eve 从保留点中均匀选出，并与所有被动 Eve 一起计入最大窃听容量。

令 A 为矩形仿真区域减去 Bob 保护区，父强度为

$$
\lambda(x)=\lambda_{\rm bg}+
\lambda_{\rm peak}\exp\!\left(-\frac{\|x-x_B\|^2}{2\sigma_c^2}\right),
\quad x\in A,
\qquad\Lambda=\int_A\lambda(x)dx.
$$

对于一组有限、非空父点，其最小标记点必然保留，因此 HCPP 非空当且仅当父 PPP 非空。强制非空的概率为

$$
p_+=1-e^{-\Lambda}.
$$

令 r_h 为硬核半径，

$$
M(x)=\int_{A\cap B(x,r_h)}\lambda(y)dy.
$$

标记 u∈[0,1] 的点保留概率为 exp[-uM(x)]，积分得到未条件化的 HCPP 一阶强度

$$
\rho_H(x)=\lambda(x)\int_0^1e^{-uM(x)}du
=\lambda(x)\frac{1-e^{-M(x)}}{M(x)}.
$$

M=0 时取连续极限1。条件非空强度为 ρ_H/p_+。当前风险图 `system_env.py:748-769` 在网格上近似了这一保留强度，但这并不等价于获得完整 SOP 或点过程 PGFL。

## 7. SOP 的空间概率表达和可计算上界

以下需要 P_J=0，冻结策略仅依赖合法侧公共信息，与本次实际 Eve 点配置无关；条件于共享合法状态 Z（含 H、Bob CSI、波束及执行 RIS 动作）后，各 Eve 的小尺度衰落标记独立。当前主动/被动 Eve 在零干扰、相同噪声下有同样的解码模型。

定义

$$
t(Z)=\frac{1+\gamma_B(Z)}{2^{R_0}}-1,\qquad
q_x(t\mid Z)=\Pr(\gamma_e(x)>t\mid x,Z).
$$

若 t<0，即 Bob 自身频谱效率已经低于 R_0，中断概率为1。以下只讨论 t≥0。

### 7.1 纯 PPP 的 PGFL

给定父 PPP 点配置，无任何 Eve 超门限的概率为 ∏_x(1-q_x)。PPP 的 PGFL 给出

$$
G_P[1-q]=\exp\!\left[-\int_A\lambda(x)q_x(t\mid Z)dx\right].
$$

记 I(t|Z)=∫_A λ(x)q_x(t|Z)dx。扣除空点集，再条件于非空，可得父 PPP 的精确条件中断概率

$$
\boxed{p_P(t\mid Z,N_P>0)=
\frac{1-e^{-I(t\mid Z)}}{1-e^{-\Lambda}}.}
$$

这是假设独立标记成立时的精确**积分表达**；若 q_x 没有闭式，它仍然不是简单闭式 SOP。没有强制非空的另一种 PPP 仿真，不能照搬分母；还须明确空 Eve 时的保密指标约定。

### 7.2 实际 HCPP 的表达

定义未条件化 HCPP 的 PGFL

$$
G_H[f]=\mathbb E\left[\prod_{x\in\Phi_H}f(x)\right].
$$

则实际条件非空的中断概率为

$$
\boxed{p_H(t\mid Z,N_H>0)=\frac{1-G_H[1-q]}{p_+}.}
$$

这是严格表达，但 HCPP 的 G_H 不是 PPP 的指数式。仅将 λ 换成“有效密度”ρ_H，不能获得精确结果。

同一个父 PPP 下有 Φ_H⊆Φ_P，保留相同位置的同一衰落标记，可逐样本比较最大 Eve SINR。因此

$$
\boxed{p_H(t\mid Z,N_H>0)\le
\frac{1-e^{-I(t\mid Z)}}{p_+}.}
$$

也可以对危险 Eve 数量使用 Campbell 公式和并集界：

$$
\boxed{p_H(t\mid Z,N_H>0)\le
\min\left\{1,\frac{\int_A\rho_H(x)q_x(t\mid Z)dx}{p_+}\right\}.}
$$

对 Z 取期望，并在 t<0 的状态下取1，就得到总体 SOP 或其上界。开启干扰后，选出的 active Eve、干扰方向、Bob SINR 与整个点配置耦合，上述简单独立标记及父集比较不能不加证明地继续使用。

## 8. Eve 超门限概率不是可直接套用的指数分布

工程 `channel_model.py:288-305,632-670` 使用

$$
g_{en}=\sqrt{\beta_e(x)}\sqrt{X_n}\,e^{j\psi_n},\quad
X_n\sim\mathrm{Gamma}(m,\Omega/m),\quad
\psi_n\sim U[-\pi,\pi],
$$

其中确定性阵列/路径相位可吸收到 ψ_n 中。条件于 Z，精确尾概率为

$$
q_x(t|Z)=\mathbb E_{g_e}\!\left[
\mathbf1\left\{
\left|h_e(x)w+\sum_n\phi_nv_ng_{en}\right|^2
>t\left(\sigma_E^2+\sigma_R^2\sum_nm_n\rho_n^2|g_{en}|^2\right)
\right\}\right].
$$

这是 Gamma 幅度与均匀相位上的多维积分。分子与有源 RIS 噪声分母含同一组幅度，不能未经推导把 SINR 宣称为 Gamma、指数或 Rice 分布。

若仅需要严格但较松的可计算界，先丢弃非负 RIS 噪声，再用 Markov 不等式。独立均匀相位使交叉项期望为零：

$$
\mathbb E[|s_e|^2\mid x,Z]=
|h_e(x)w|^2+\beta_e(x)\Omega\sum_n\rho_n^2|v_n|^2.
$$
$$
q_x(t|Z)\le\min\left\{1,
\frac{|h_e(x)w|^2+\beta_e(x)\Omega\sum_n\rho_n^2|v_n|^2}
{t\sigma_E^2}\right\},\qquad t>0.
$$

可将这个 q 上界代入上一节的 SOP 上界，得到不依赖瞬时 Eve CSI 的空间积分上界；通常保守，是否有用必须数值检查。

若论文需要更紧的半解析近似，可以在大 N、无单元占主导时用复高斯 CLT 近似 RIS 叠加，并把小量 RIS 噪声用其均值近似。令

$$
V_e=\beta_e\Omega\sum_n\rho_n^2|v_n|^2,\quad
\bar Q_e=\sigma_E^2+\sigma_R^2\beta_e\Omega\sum_nm_n\rho_n^2,
$$

则可近似使用

$$
q_x(t|Z)\approx Q_1\!\left(
\sqrt{2|h_e(x)w|^2/V_e},\sqrt{2t\bar Q_e/V_e}
\right).
$$

这是近似而非严格上界；V_e=0 应单独用直连确定性门限判断。在稀有中断尾部，必须用足量 Monte Carlo 验证，不能只比较平均容量。

## 9. 相位为何主要增强 Bob，而没有系统性抑制 Eve

在上一节模型中，对于只依赖合法侧信息的任意相位选择，Eve 的各单元相位仍为独立均匀相位：

$$
\{g_{en}e^{j\theta_n}\}_n\overset d=\{g_{en}\}_n.
$$

有源 RIS 噪声又只依赖 |g_en|²，因此条件于 H、w、模式、振幅和 Eve 位置时，Eve SINR 的分布对 RIS 相位不变。保持 w、模式、振幅不变，相位优化不能在统计意义上系统降低最大 Eve 容量；单次信道样本仍可能上升或下降。这一结论不否认波束、位置、模式、干扰或 Eve 信息辅助设计的作用。

Bob 则通过自身 CSI 把 N 个反射项对齐。若每单元贡献近似同量级且增益不因总功率约束下降，Bob 相干反射信号功率约按 N² 增长，而未知 Eve 的平均反射信号功率按 N 增长。RIS 噪声也随有源单元数增加；固定孔径、空间相关性、功率投影或大 N 饱和时，不能无限外推 N²。

这解释了既有仿真中增大 N 主要提升 Bob、最大 Eve 容量变化很小的现象。论文应把“提升保密性”与“实际降低 Eve 容量”分别给出证据。

## 10. 离线数值核验

本目录 `verify_bounds.py` 在不加载 RL 网络、不训练的条件下，对 N=64/96/112/128/256 各采样10个几何布局，每个10个衰落时隙。每个信道比较 MRT＋全有源零残差，以及随机可行波束、组模式和±30°组残差，共1000组执行动作。G统一为16；4×4风险网格仅用于加速未使用风险图的解析参考，未改变物理信道参数。

全部满足 D_L≤D≤D_U（上界是固定动作三角界，见核验脚本，并非重新计算旧文档的全动作上界）。最大上界偏差约3.1×10^(-16)，属于浮点误差。应用源文件 SHA-256 核对一致；未操作当前训练进程。

下表仅为每个 N 的100个 **MRT＋全有源相干参考样本** 汇总，容量单位 bit/s/Hz。下界采用整个±30°残差域，参考动作实际残差为0，因此会保守。

| N | 平均 D_L | 平均实际 D | 实际经验 SOP | 由 D_L 得到的经验频率上界 |
|---|---:|---:|---:|---:|
| 64 | 0.04973 | 0.05944 | 1.00 | 1.00 |
| 96 | 0.07938 | 0.09514 | 0.88 | 1.00 |
| 112 | 0.09423 | 0.11312 | 0.00 | 0.92 |
| 128 | 0.11028 | 0.13246 | 0.00 | 0.02 |
| 256 | 0.26044 | 0.31473 | 0.00 | 0.00 |

100个样本不足以估计罕见中断；112单元此前更大5000样本参考评估曾有6次中断。以上均非新 MAPPO 训练结果，经验上界不是总体 SOP 的统计置信上界。平均 D_L>R_0 也不意味着每个时隙达标，例如128单元仍有2个样本的下界未达标。

## 11. 论文中的建议表述

1. 将单流 Gaussian 频谱效率、安全速率、目标门限与经验 SOP 明确定义，并说明 Eve 非串谋。
2. 给出“解析基相位＋组残差”的 Bob 幅度下界和保密差下界定理，说明使用执行振幅。
3. 给出 SOP 的事件包含界；用同一批冻结策略样本绘制界与实际曲线，报告松紧程度。
4. 明确实际 Eve 是非空 Matérn-II HCPP；纯 PPP PGFL 可作分析对照或上述父 PPP 上界，不能当作当前模型精确公式。
5. 区分严格界、空间数值积分、CLT 近似、有限样本频率，不把平均容量、最小观测容量或零中断样本称为全衰落保证。
6. 当前模型可论证“增强 Bob 并降低 SOP”，不能仅凭 RIS 相位学习声称“统计上系统性压低无 CSI Eve 容量”。

## 12. 外部理论参考

- Last and Penrose, *Lectures on the Poisson Process*：PPP Laplace/概率生成泛函、Campbell 公式。作者公开教材：https://stoch.math.kit.edu/img/Last/lastpenrose2017.pdf
- Teichmann et al., *Generalizations of Matérn's hard-core point processes*：依赖稀疏化与硬核点过程的一、二阶性质。https://arxiv.org/abs/1209.2566
- Shi et al., *On Secrecy Performance of RIS-Assisted MISO Systems over Rician Channels with Spatially Random Eavesdroppers*：PPP 下 RIS 保密分布分析的相关工作；其信道假设与本工程不同，不能直接移植其闭式公式。https://arxiv.org/abs/2312.16814

上面的工程专用不等式和 HCPP 父集比较是依据当前代码逐步推导，不声称为上述论文中原样给出的结果。
