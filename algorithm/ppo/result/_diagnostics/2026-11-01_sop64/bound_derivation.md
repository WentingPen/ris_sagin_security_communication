# 64 单元 SOP 诊断：抽样时隙的保密速率乐观上界

这里的上界仅用于离线可达性核查，不作为 Actor 的输入。评估对象为各次运行最终 checkpoint；每个 N 使用 10 个相同的几何种子，每个几何取 50 个衰落时隙。不同策略在同一时隙使用相同 CSI，不同 N 的衰落实现不要求逐元素一致。

条件：一个 Bob、一个数据流、干扰发射功率为零；实际代码满足这些条件。接收机噪声和有源 RIS 输入噪声均按原环境设置，不修改训练参数。

## 构造

记卫星功率上限为 P，RIS 最大振幅为 a，卫星到 RIS 的第 n 行信道为 H_n。Bob 和 Eve e 的直达信道分别为 h_B、h_e，RIS 到接收端的单元信道分别为 g_Bn、g_en。

对任意可行波束 w，令 d = |h_B w|，则：

$$
0\le d\le \sqrt{P}\|h_B\|,\qquad
u_B=\sqrt{P}a\sum_n|g_{Bn}|\|H_n\|.
$$

放松 Bob 的有源 RIS 噪声，并允许所有单元取最大振幅和任意相位，得到：

$$
C_B\le B(d)
=\log_2\left(1+\frac{(d+u_B)^2}{\sigma_B^2}\right).
$$

将 Eve 直达信道正交分解为：

$$
h_e=\alpha_e h_B+\epsilon_e,\qquad
\alpha_e=\frac{h_e h_B^H}{\|h_B\|^2}.
$$

由反三角不等式和 Cauchy–Schwarz 不等式，定义：

$$
u_e=\sqrt{P}\left(\|\epsilon_e\|+
a\sum_n|g_{en}|\|H_n\|\right).
$$

其接收信号幅度下界为 [|\alpha_e|d-u_e]^+。有源 RIS 噪声上界为：

$$
v_e=\sigma_E^2+\sigma_{\mathrm{RIS}}^2a^2\sum_n|g_{en}|^2.
$$

因此 Eve 容量有保守下界：

$$
C_e\ge L_e(d)=
\log_2\left(1+\frac{([|\alpha_e|d-u_e]^+)^2}{v_e}\right).
$$

于是任何波束、单元模式和相位的原始保密差都满足：

$$
C_B-\max_e C_e
\le
\max_{0\le d\le\sqrt{P}\|h_B\|}
\left(B(d)-\max_e L_e(d)\right).
$$

该界还忽略了 RIS 输出 RF 功率约束和分组相位残差范围，因此比实际动作可行域更宽松。

## 数值最大化保持上界方向

将 d 的区间分成 2000 个小区间。B 和各 L_e 都随 d 单调不减。因此在区间 [d_i,d_(i+1)] 内采用：

$$
B(d_{i+1})-\max_e L_e(d_i)
$$

作为整个小区间的上界，再取所有小区间的最大值。这里不是简单取离散采样点的最大值，避免将网格最大值误称为连续域上界。

## 本次结果与限制

- N=64、P=25 W、a=8：500 个抽样时隙的上界最大值约 0.07175 bit/s/Hz，均低于目标 0.1。
- 已逐时隙检查所有对照策略的原始保密差不超过对应上界。
- 这说明这些抽样信道下，即使允许比当前策略更宽的动作域，仍不能达标。
- 结论不推广为所有随机衰落实现下的严格全局不可达性。Nakagami/Rician 衰落有尾部，有限样本 SOP=1 仍是经验结果。

原始时隙数据见同目录 slot_metrics.csv，汇总见 diagnostic_report.json。
