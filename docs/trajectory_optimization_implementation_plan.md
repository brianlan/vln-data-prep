# 基于 A* 初始路径的麦克纳姆底盘轨迹优化实施计划

## 1. 文档目的

本文定义基于 A* 初始路径的离线轨迹优化实施路线，覆盖：

- 第一阶段的最小可行实现；
- 麦克纳姆底盘运动学诊断器；
- 当诊断表明轮速超限过多时触发的第二阶段；
- 每个阶段的输入、输出、验收标准和明确非目标。

本文只规划工作。轨迹优化作为独立后处理步骤运行，不覆盖现有 SAGE3D 轨迹产物。允许对原生成器做窄范围修改，额外保存优化所需的只读中间产物，但不改变已有产物的字段和语义。

计划基线：

- Git commit：`f8548ca6ac7bf9b9f9d16ff7e49cf8e3cc8c5a63`
- 工作分支：`trajectory-optimization-plan`
- 当前实现入口：`generate_sage3d_trajectories.py`

## 2. 总体结论与实施原则

总体技术路线可行：

```text
膨胀栅格上的 A*
    ↓ 选择绕障拓扑并提供初始路径
五次 clamped B-spline
    ↓ 联合表示 x、y、yaw
控制点与总时间 T 的约束优化
    ↓
独立高密度验证
    ↓
固定控制周期的时序轨迹
```

A* 不负责直接生成完整时序状态。它只提供离散二维路径和一个合理的绕障拓扑。B-spline 优化器负责在该拓扑附近改善几何形状、yaw 和时间尺度。独立验证器负责确认最终输出，而不能只信任求解器的成功状态。

实施遵循以下原则：

1. 先验证数学内核，再接入真实地图。
2. 先支持静止到静止，再支持非零边界速度。
3. 第一阶段使用独立输出目录，不覆盖原始轨迹；原生成器只在现有 episode NPZ 中新增 `astar_path_pixels`，并在现有 manifest 中补充地图变换字段。
4. 碰撞、安全边界和导数上限属于验收条件，不用大权重近似替代。
5. 先用当前已安装的 NumPy 和 SciPy；没有基准证据前不引入 CasADi/IPOPT。
6. 当前 `safe` 栅格已经包含机器人半径和 safety margin，不得再次重复膨胀。

## 3. 当前代码基线

当前 SAGE3D 轨迹生成器具有以下行为：

- 在已经过机器人半径和安全余量过滤的栅格上运行 A*；
- 使用 clearance 代价让 A* 倾向于远离障碍物；
- 对 A* 路径做可见性简化、三次 spline 或局部 Bézier 平滑；
- 按固定空间距离 `frame_spacing` 重采样；
- 通过相邻位置梯度得到 yaw；
- 输出位置、姿态矩阵、相机位置、yaw 和 PointGoal；
- 不输出时间戳、世界速度、角速度、加速度或 jerk。
- 不保存原始 A* 完整点列；当前输出已经包含 `map/safe_mask.png`、`map/esdf.npy` 以及 manifest 中的地图 shape 和分辨率，但 manifest 还缺少 `MapTransform` 的 `lower_x`、`lower_y` 和坐标变换约定。

现有 episode NPZ 中的 `points` 可以作为由 A* 派生的几何参考路径，现有地图文件可以直接用于碰撞验证；仍需保存原始 A* 点列并补全地图变换，才能在独立优化进程中恢复权威拓扑和像素/世界坐标关系。新的优化器不是现有 `smooth_path()` 的局部替换，而是读取扩展后原生成结果的独立后处理器。

## 4. 全局范围与阶段边界

### 4.1 第一阶段假设

第一阶段固定为：

- 离线规划；
- 静态障碍物；
- 圆形底盘碰撞 footprint；
- 任意起终位置和 yaw；
- 起终世界平移速度为零；
- 起终 yaw rate 为零；
- 起终加速度只要求不超限，不强制为零；
- `vx`、`vy` 在优化和内部输出中均表示世界坐标系速度；
- yaw 在优化内部是 unwrap 后的连续实数；
- 使用一个全局总时间 `T`，不使用独立 span 时长；
- 最终轨迹按固定控制周期 `dt` 采样。

### 4.2 第一阶段明确非目标

第一阶段不包含：

- 非零起终速度或紧急避障数据生成；
- kinodynamic A*；
- 多条 A* 拓扑候选；
- 正式凸安全走廊或 Bézier extraction；
- 每个 span 的独立时间变量；
- 完整 jerk-limited S-curve；
- 轮速或轮加速度主优化硬约束；
- 电机电流、转矩、轮胎力或打滑模型；
- 在线重规划、动态障碍物和新旧轨迹拼接；
- 改变现有 episode NPZ、manifest、渲染或 LeRobot 已有字段的值和语义；允许在 episode NPZ 和 manifest 中增加不会被现有消费者误读的新字段。

## 5. 第一阶段目标

第一阶段要证明：

> 给定原生成器扩展后的轨迹目录及同级地图数据，独立优化脚本能够稳定产生一条具有真实时间含义的静止到静止五次 B-spline 轨迹；所有被接受的轨迹都通过独立碰撞和导数限制验证。

第一阶段的输出是内部研究产物，不直接替换现有专家轨迹。

## 6. 第一阶段实施计划

### 6.1 工作包一：冻结接口、单位和坐标系

独立脚本从原始轨迹目录读取数据，并将每条 episode 转换为以下内部问题对象。原生成器与优化器之间不通过 Python 函数调用传递运行时对象。

定义最小输入对象：

```python
TrajectoryProblem
├── astar_path_xy             # 世界系离散 A* 路径，单位 m
├── safe_mask                 # 已膨胀且已做相机过滤的安全栅格
├── clearance_m               # 原始自由空间到障碍物的距离，单位 m
├── map_transform             # 像素与世界坐标变换
├── start_pose                # [x, y, yaw]
├── goal_pose                 # [x, y, yaw]
├── required_clearance_m      # robot_radius + safety_margin
└── limits                    # 时间和导数限制
```

定义内部输出对象：

```python
TimedTrajectory
├── time                      # [K]，严格递增，单位 s
├── position_world            # [K, 2]，单位 m
├── yaw_unwrapped             # [K]，单位 rad
├── velocity_world            # [K, 2]，单位 m/s
├── acceleration_world        # [K, 2]，单位 m/s^2
├── jerk_world                # [K, 2]，单位 m/s^3
├── yaw_rate                  # [K]，单位 rad/s
├── yaw_acceleration          # [K]，单位 rad/s^2
├── yaw_jerk                  # [K]，单位 rad/s^3
├── total_time                # T，单位 s
└── solver_metadata
```

本轮暂定控制周期：

$$
dt=0.1\ \mathrm{s}.
$$

总时间上下界以 episode NPZ 中 `astar_path_pixels` 转换得到的原始 A* 世界坐标折线长度为基准：

$$
S_{A*}=\sum_i\|Q_{i+1}^{W}-Q_i^{W}\|_2,
$$

$$
T_{min}^{policy}=1.0\ \mathrm{s/m}\cdot S_{A*},\qquad
T_{max}^{policy}=3.0\ \mathrm{s/m}\cdot S_{A*}.
$$

系数具有 `s/m` 单位。这组边界相当于把相对于原始 A* 长度的平均速度限制在约 `0.33–1.0 m/s`。`S_A*` 在一次优化中保持不变，不改用优化后曲线长度，否则时间边界会反过来随优化变量变化。

为了与固定控制周期一致，NLP 使用的实际边界为：

$$
T_{min}=\left\lceil\frac{T_{min}^{policy}}{dt}\right\rceil dt,\qquad
T_{max}=\left\lfloor\frac{T_{max}^{policy}}{dt}\right\rfloor dt.
$$

这两个边界是数据时长策略，不是动力学可行性的充分条件。起终静止、转弯、yaw 和 jerk 是否可行仍由硬约束决定；若在 `T_max` 内没有可行解，则记录 `TIME_LIMIT_FAILED`，不自动放宽到 `3S_A*` 之外。

当前默认 A* 路径长度范围为 3–15 m，因此上述规则对应 3–45 s，以及在 `dt=0.1 s` 下约 31–451 个含首尾样本。已确认该最大 episode 时长和帧数对后续数据量可以接受；首版不另设全局最大时长或最大帧数。

仍必须在实现前确认：

- 平移速度上限暂定为用户已提供的 `v_max=0.6 m/s`，其权威来源仍待记录；
- 平移加速度和 jerk 限值；
- yaw rate、yaw acceleration 和 yaw jerk 限值；
- 数值验收容差和安全余量；
- 限值来自硬件、仿真器还是数据质量要求。

> **【需要用户输入】** 请补充 `v_max=0.6 m/s` 的权威来源，并逐步提供其余尚未确认参数的期望数值、单位和来源；若某项没有硬件依据，也需要明确它是仿真限制还是人为的数据质量标准。没有这些输入时可以实现参数化代码，但不能宣称输出具有真实动力学可执行性。

所有数值集中放入带 schema version 的 `configs/trajectory_optimization_v1.json`，首版 schema 固定为 `vln_data_prep.trajectory_optimization_config.v1`。配置至少分为 `timing`、`translation_limits`、`yaw_limits`、`spline`、`initialization`、`objective`、`constraints`、`solver`、`validation` 和 `mecanum`。CLI 只提供输入目录、输出目录、配置路径和少数显式覆盖项；不得把 limits、目标权重和求解器参数分散为大量无版本 CLI 默认值。优化输出必须保存完整的 effective config 及其 SHA-256。

平移速度、加速度和 jerk 首版统一使用二维欧氏范数：

$$
\|v^{W}\|_2\le v_{max},\qquad
\|a^{W}\|_2\le a_{max},\qquad
\|j^{W}\|_2\le j_{max}.
$$

yaw 导数使用绝对值。首版不同时增加逐轴限制，避免两套语义互相覆盖。

已确认第一阶段对世界坐标系下的平移速度、加速度和 jerk 使用二维欧氏范数上限，对 yaw 各阶导数使用绝对值上限，不增加世界坐标逐轴约束。若后续真实控制器给出独立的前向/侧向上限，应在机器人 body frame 中表达，或通过麦克纳姆轮速约束处理，不能直接当作世界坐标逐轴上限。

起终 yaw 默认读取原 episode 的 `yaw[0]` 和完整 yaw 序列 unwrap 后的 `yaw[-1]`，不再重复保存默认边界 yaw。优化 CLI 可以覆盖，但 effective value 必须写入输出 manifest。

已确认第一阶段默认使用原 episode 的 `yaw[0]` 和完整 yaw 序列 unwrap 后的 `yaw[-1]`，不重新随机采样起终 yaw。优化器仍须支持通过显式 CLI 参数覆盖边界 yaw；只有用户主动指定时才覆盖，且输出 manifest 必须保存最终采用的 effective value。

总时间 `T` 是连续优化变量，最终输出时间必须落在固定控制周期上。首版采用：

1. 连续求解得到 `T_continuous`；
2. 令 `N=ceil(T_continuous / dt)`；
3. 设置 `T_output=N*dt`；
4. 在 `[0,T_output]` 上以严格固定 `dt` 重新求值；
5. 重新计算全部导数、目标函数、碰撞和运动学诊断。

不使用短于 `dt` 的最后一个间隔，不用 `linspace` 改变实际周期，也不把 `N` 放入第一阶段 NLP 形成混合整数问题。由于只向上取整时间，在固定几何曲线下导数上限不会恶化，但时间目标会改变。

NLP 强制 `T_continuous <= T_max`，其中 `T_max` 已经是 `dt` 的整数倍，因此向上取整后的 `T_output` 仍不得超过 `T_max`；若因数值容差越界则拒绝结果。

已确认采用“向上取整到 `N*dt` 后重新验收”：`T_output` 是最终权威时长，第 7 节的目标比较、约束验收和保存结果均以重新时间参数化后的输出轨迹为准，而不是以 `T_continuous` 为准。独立验证器仍在控制帧之间进行高密度采样，不会只检查每隔 `dt` 的输出点。

验证：接口测试应拒绝量纲不合法、非有限值、重复路径点导致的零长度路径和已经越界的起终点。

### 6.2 工作包二：五次 B-spline 数学内核

使用固定 open-uniform clamped quintic B-spline：

$$
q(u)=\sum_i N_{i,5}(u)P_i,\qquad u\in[0,1].
$$

真实时间映射：

$$
u=t/T.
$$

必须实现并测试：

- clamped knot vector；
- 基函数和曲线求值；
- 一、二、三阶参数导数；
- 导数控制点；
- 参数导数到真实时间导数的转换；
- 起终位置和零速度边界；
- 平移与 yaw 的 jerk 平方积分；
- yaw unwrap 和最终 wrap。

真实时间缩放必须满足：

$$
\dot q_t=\frac{1}{T}q'_u,
\qquad
\ddot q_t=\frac{1}{T^2}q''_u,
\qquad
q^{(3)}_t=\frac{1}{T^3}q^{(3)}_u.
$$

jerk 平方积分必须满足：

$$
\int_0^T\|q^{(3)}_t(t)\|^2dt
=
\frac{1}{T^5}
\int_0^1\|q^{(3)}_u(u)\|^2du.
$$

验证：

- 基函数非负且和为一；
- clamped 曲线精确经过端点；
- 零速度控制点关系精确成立；
- 解析导数与有限差分一致；
- 把 `T` 放大两倍后，一、二、三阶导数分别按 `1/2`、`1/4`、`1/8` 缩放；
- jerk 积分按 `1/32` 缩放；
- `170° → -170°` 的最短旋转内部表示为 `170° → 190°`。

### 6.3 工作包三：A* 参考曲线和初始控制点

对 A* 点列计算累计弧长：

$$
s_0=0,\qquad
s_i=s_{i-1}+\|Q_i-Q_{i-1}\|.
$$

原始 A* 点列负责固定绕障拓扑并初始化控制点；现有 episode 的平滑 `points` 构造 `r(s)`，用于 `J_ref`。诊断同时记录优化曲线到原始 A* 折线和到平滑参考曲线的距离，避免混淆二者职责。

按路径长度动态确定控制点数量：

$$
N_{ctrl}
=
\operatorname{clip}\left(
\left\lceil\frac{L}{\ell_{ctrl}}\right\rceil+p,
N_{min},N_{max}
\right),\qquad p=5.
$$

首版建议默认 `target_control_spacing_m=0.5`、`min_control_points=8`、`max_control_points=64`。五次 spline 的数学最低控制点数为 6，但默认 8 为端点零速度关系之外保留更多内部自由度。三个值都写入版本化配置，并通过固定基准调整，不在代码中散落。

平移控制点初始化使用一个小型约束最小二乘或 QP：

$$
\min_P
\sum_i\|P_i-\widetilde P_i\|^2
+
\lambda_{init}
\sum_i\|P_{i+1}-2P_i+P_{i-1}\|^2.
$$

其中 `P_tilde` 来自按弧长均匀采样的 A* 参考路径。初始化平滑项只用于得到良好初值，不代表真实物理加速度。

初始化 QP 固定四个端点控制点后消元，使用 `numpy.linalg.solve` 求解维度至多为 60、包含 `x/y` 两个右端项的稠密线性系统，不引入通用 QP solver。建议默认 `lambda_init=1.0`、`gamma=1.2`；位置拟合项和二阶差分平滑项具有相同的长度平方量纲，因此 `lambda_init` 为无量纲值。

yaw 初始化不采用“起终 yaw 全路径线性插值”与路径切线的全局混合。该方法会让 U 型路径在第一段过早旋转，而且把 `yaw_tangent_weight` 误解为 `[0,1]` 混合比例；大于 1 时还会发生外推。

首版使用现有 episode 平滑 `points` 及其完整 unwrap `yaw` 构造局部切线参考，并求解约束平滑问题：

$$
\min_{\Theta}
w_{tangent}
\sum_i(\Theta_i-\widetilde\Theta_i)^2
+
\sum_i(\Theta_{i+1}-2\Theta_i+\Theta_{i-1})^2.
$$

其中 `Theta_tilde` 按平滑参考路径的归一化弧长重采样到 yaw 控制点位置。二阶差分项的系数固定为 1，`yaw_tangent_weight=w_tangent` 表示切线参考相对于平滑项的非负权重，不再表示混合百分比，也不限制在 `[0,1]`。

yaw 初始化采用：

1. 默认边界使用现有 episode 完整 unwrap yaw 的首尾值，保留 U 型路径等情况下的连续旋转分支，不再把终点强制折回相对起点的最短全局分支；
2. 若 CLI 覆盖边界 yaw，则通过加减 `2*pi` 将覆盖值提升到最接近平滑参考对应端点的连续分支；
3. 中间控制点使用平滑路径切线作为软参考，以抑制直线段提前旋转；二阶差分项只负责抑制 yaw 突变；
4. 精确约束 `Theta[0]=Theta[1]=start_yaw`、`Theta[-2]=Theta[-1]=goal_yaw`，从而满足起终 yaw 和 yaw rate 边界。

`yaw_tangent_weight` 必须进入版本化配置和输出 manifest，不能隐藏在实现中。首版不预设权威默认值；应在直线、直角、S 形和 U 型固定案例上比较第一段朝向误差、弯道提前旋转和 yaw 导数峰值后再确定。

初始时间根据导数控制点估计，并裁剪到本节确定的时长策略范围：

$$
T_{init}=\operatorname{clip}
\left(
\gamma\max(T_v,T_a,T_j,T_{min}),
T_{min},T_{max}
\right),
\qquad \gamma>1.
$$

如果未裁剪候选值超过 `T_max`，必须在日志中记录 `initial_time_exceeds_policy_max=true`；这表示初始几何很可能无法在时长策略内满足导数限制，但最终是否无解仍由主优化器和独立验证决定。

验证：初始曲线必须通过端点、满足零边界速度，并且在进入主优化器前生成完整的碰撞与导数诊断报告。U 型案例还必须检查第一段没有系统性提前旋转，且 unwrap yaw 没有非预期的 `2*pi` 跳变。

### 6.4 工作包四：最小联合优化器

优化变量为内部平移控制点、yaw 控制点和总时间：

$$
z=[P^{xy}_{internal},\Theta_{internal},T].
$$

第一阶段目标函数仅包含：

$$
J=
w_{ref}\bar J_{ref}
+w_{j,p}\bar J_{jerk,p}
+w_{j,\theta}\bar J_{jerk,\theta}
+w_{\omega}\bar J_{yaw\ rate}
+w_T\frac{T}{T_{scale}}.
$$

每一项都必须按典型物理尺度归一化，并单独记录未加权值、归一化值和加权值。

积分项使用每个非零 knot span 上固定阶数的 Gauss-Legendre quadrature，不使用“当前有多少采样点”的简单求和。首版定义：

- `J_ref`：优化曲线与平滑参考曲线的平方距离，对归一化路径进度积分，并除以 `reference_distance_scale_m^2`；
- 平移/yaw jerk：真实时间 jerk 平方积分，分别除以对应 jerk scale 的平方和 `time_scale_s`；
- yaw rate：真实时间 yaw rate 平方的时间平均值；
- 时间项：NLP 内使用 `T_continuous / time_scale_s`；输出离散化后重新报告 `T_output / time_scale_s`。

这样增加绘图点、验证点或输出帧不会改变目标函数数值。quadrature 阶数、全部 scale 和全部权重都属于版本化配置。

第一阶段硬约束：

- 起终位置和 yaw；
- 起终世界速度和 yaw rate 为零；
- `T_min <= T <= T_max`；
- 平移速度、加速度和 jerk 上限；
- yaw rate、yaw acceleration 和 yaw jerk 上限；
- 优化配点处的地图 clearance 约束；
- 控制点相对初始解的保守 trust region。

碰撞约束使用 `clearance_m` 的双线性插值，但最终是否安全仍由独立 `safe_mask` 和高密度碰撞验证决定。第一阶段不宣称有限配点给出了连续碰撞证明。

平移导数约束首版在每个非零 knot span 的 5 个固定配点上施加；独立验证器使用不同且更密集的点。第一阶段不使用导数控制点凸包约束，因此同样不宣称连续导数上限证明。

trust region 的位置和 yaw 使用不同量纲：

- `xy` 半径根据地图分辨率和候选控制点的局部剩余 clearance 计算，并受 `trust_xy_max_m` 截断；
- yaw 使用单独的 `trust_yaw_rad`；
- 第一阶段使用固定规则，不在失败后自动扩大 trust region；
- 具体默认系数写入配置，并由合成案例和固定真实基准校准。

求解器顺序：

1. 使用 SciPy SLSQP 快速验证变量、目标和约束方向；
2. 若基准显示 SLSQP 不稳定，再比较 `trust-constr`；
3. 只有在固定基准证明现有求解器是主要瓶颈后，才单独评估 CasADi/IPOPT。

SLSQP 首版建议配置 `ftol=1e-8`、`maxiter=1000`、`episode_timeout_s=60`。超时通过 callback 检查单调时钟并中止本 episode。求解成功必须同时满足：

- `result.success` 为真；
- 目标和变量均有限；
- 独立重算的最大等式/不等式违反不超过各自容差；
- 最终高密度验证通过。

不能只看 `result.success`。若固定真实基准的端到端通过率低于 90%，或者超过 5% 的结构有效输入因迭代上限、线搜索或数值失败而失败，则触发 `trust-constr` 对照；不在单 episode 内静默切换求解器。

### 6.5 工作包五：独立高密度验证器

验证器必须与目标函数和求解器成功状态解耦。它至少检查：

- 时间有限、严格递增；
- 所有数组长度一致；
- 无 NaN 或 Inf；
- 起终位置、yaw、世界速度和 yaw rate；
- yaw unwrap 连续；
- 每个 knot span 的高密度 footprint 碰撞；
- 相机中心到 3D collision mesh 的 clearance；
- 平移速度、加速度和 jerk；
- yaw rate、yaw acceleration 和 yaw jerk；
- 求解器报告的最大约束违反；
- 路径长度、总时间和目标函数各分项。

任何硬约束验证失败都必须拒绝轨迹，而不是仅记录 warning。

碰撞语义固定如下：

- footprint 验证只查询轨迹中心是否位于已经完成机器人半径膨胀及相机栅格过滤的 `safe_mask`；
- 不再用圆形 footprint 对 `safe_mask` 做第二次膨胀；
- 地图越界一律视为碰撞；
- 双线性 clearance 查询要求四个邻接 cell 都在数组内，否则视为不可行；
- 恰落在 cell 边界时使用统一的世界到连续像素坐标公式，不调用带 `round()` 的离散查询作为优化约束；
- 优化阶段允许某个候选满足二维 `clearance_m` 但进入相机不安全 cell，然而最终 `safe_mask` 或 3D mesh 验证会拒绝它；这种情况记录为 `CAMERA_CLEARANCE_FAILED`，不能标记为可执行。

独立优化器取得 3D collision mesh 的规则为：

1. 默认读取原 manifest 的 `collision_usd`；
2. CLI 可用 `--collision-usd` 覆盖不可移植的原路径；
3. 原 manifest 新增生成时 collision USD 文件的字节 SHA-256 和文件大小；
4. 默认路径或覆盖路径必须通过摘要校验；
5. 文件缺失或摘要不匹配时结构性失败，不跳过 3D 验证。

高密度验证对每个非零 span 的采样数取以下三者最大值：

```text
validation_points_per_span
ceil(estimated_span_length / (validation_space_fraction * map_resolution)) + 1
ceil(estimated_span_duration / (validation_time_fraction * dt)) + 1
```

首版建议 `validation_points_per_span=50`、`validation_space_fraction=0.25`、`validation_time_fraction=0.25`。边界状态、碰撞、导数和求解器可行性分别使用独立容差，不设置一个全局 epsilon。实际容差属于第 6.1 节的用户输入和版本化配置。

失败原因至少分为：

```text
INVALID_INPUT
INITIALIZATION_INFEASIBLE
SOLVER_DID_NOT_CONVERGE
COLLISION_VALIDATION_FAILED
BOUNDARY_STATE_FAILED
TRANSLATION_LIMIT_FAILED
YAW_LIMIT_FAILED
TIME_LIMIT_FAILED
CAMERA_CLEARANCE_FAILED
WHEEL_SPEED_FAILED
WHEEL_ACCEL_FAILED
NUMERICAL_FAILURE
```

一条轨迹可以保存多个 violation。用于聚合统计的 `primary_failure_reason` 使用固定优先级：结构性输入错误、数值异常、边界状态、碰撞/相机 clearance、平移/yaw 导数、时间、轮速/轮加速度、求解器状态。具体 `violations` 列表保留全部原因，不能因主原因而丢失其他问题。

### 6.6 工作包六：固定验证集和基准报告

合成案例至少包含：

- 无障碍直线；
- 单个直角转弯；
- S 形路径；
- 狭窄通道；
- 贴障转弯；
- 起终 yaw 与运动方向不一致；
- yaw 跨越 `±pi`；
- 过短时间上限导致的预期无解；
- 不可能满足 clearance 的预期无解。

真实案例固定一组 A* 路径，不在每次基准运行时重新随机抽样。建议第一轮至少包含 50 条路径，并覆盖不少于 3 个具有不同几何特征的场景。

> **【需要用户输入】** 请指定首批 scene ID、episode ID 或可生成它们的固定输入目录，以及基准机器。还需确认扩展后的 trajectory fixture 是提交小型样例到仓库，还是通过外部 artifact 加载。计划默认记录固定 seed、运行 3 次，并把基准机器的 Python、CPU、内存和求解器版本写入报告。

真实通过率的分母是“全部结构有效的固定输入”。初始化、求解和最终验证失败都计入失败；只有缺文件、schema 错误或摘要不匹配等结构无效输入先使整个基准无效，而不是从分母中删除。

基准报告至少包含：

- 初始化成功率；
- 求解器成功率；
- 独立验证通过率；
- 各失败原因计数；
- p50、p95 求解时间；
- p50、p95 总时间和路径长度；
- 优化前后各目标项；
- clearance 和各导数最大利用率；
- 固定 seed 重复运行差异。

## 7. 第一阶段验收门槛

第一阶段完成需要同时满足：

1. 所有数学单元测试通过。
2. 所有确定性合成可行案例通过独立验证。
3. 所有预期无解案例被明确分类，而不是输出违规轨迹。
4. 真实固定路径集的端到端通过率不低于 90%。
5. 所有被接受轨迹的碰撞和导数硬约束违反数为零。
6. 固定输入在同一基准机器上运行 3 次；episode 状态、失败原因和数组 shape 必须完全一致，浮点数组及目标值在配置规定的 `rtol/atol` 内一致。
7. 对同一 effective config 下通过全部硬约束的完整初始轨迹，最终 `T_output` 对应的归一化总目标不高于初始轨迹目标加数值容差。若初始解不可行，必须单独报告，不能把它作为该项的可比基线。
8. 生成完整的麦克纳姆运动学诊断报告。
9. 现有 episode NPZ 数组和 manifest 字段保持原有值与语义；新增 `astar_path_pixels` key 和 manifest 元数据不得改变渲染、打包或 LeRobot 输出，现有消费者必须忽略未知 key。

90% 是研究原型门槛，不是生产发布门槛。生产集成前需要单独制定更高的成功率和吞吐量目标。

## 8. 独立脚本、扩展输入与代码组织

### 8.1 两步运行边界

轨迹生成和轨迹优化是两个独立进程：

```text
generate_sage3d_trajectories.py
    ├── 保持当前 A*、几何平滑和原始产物输出
    ├── 在现有 episode NPZ 中新增 astar_path_pixels
    └── 在现有 manifest 中补充 MapTransform 元数据

optimize_sage3d_trajectories.py
    ├── 读取扩展后的 trajectory directory 及其同级 map 目录
    ├── 执行 B-spline 优化、验证和运动学诊断
    └── 写入另一个 optimized trajectory directory
```

`optimize_sage3d_trajectories.py` 不导入或调用 `generate_sage3d_trajectories.py`，不重新运行 A*，也不原地修改输入目录。若优化失败，原始生成结果保持可用。

### 8.2 原生成器需要增加的最小字段

当前 episode 的 `points` 和 `yaw` 已经提供由 A* 派生的平滑参考轨迹，场景 `map/` 目录已经包含 `safe_mask.png` 和 `esdf.npy`，manifest 已经包含地图 shape、分辨率、安全参数、scene/seed 和 collision USD 路径。第一阶段直接复用这些现有数据，不创建 `context.json`、`map.npz` 或 `optimization_inputs/`，也不重复保存默认边界 yaw。

扩展后的目录保持为：

```text
<scene-dir>/
├── map/
│   ├── safe_mask.png                     # 直接复用
│   └── esdf.npy                          # 直接复用，语义为 clearance_m
└── trajectories/
    ├── trajectory_manifest.json          # 增加 MapTransform/mesh 摘要字段
    ├── episode_000000.npz                 # 增加 astar_path_pixels key
    └── episode_000001.npz
```

`trajectory_manifest.json` 新增：

- `map.lower_x`、`map.lower_y`；
- `map.safe_mask_semantics="robot_inflated_and_camera_filtered_v1"`；
- `collision_usd_size_bytes` 和 `collision_usd_sha256`。

现有 `map.shape`、`map.scale_m_per_pixel`、`map.required_path_clearance_m`、`map.camera_collision_filter`、scene ID、seed 和 `collision_usd` 继续作为权威字段，不另存副本。`MapTransform` 的 `height`、`width` 分别取 `map.shape[0]`、`map.shape[1]`，`scale` 取 `map.scale_m_per_pixel`；加上新增的 `lower_x`、`lower_y` 后即可完整重建。优化器用该变换在 `[row,col]` 和世界 `[x,y]` 米制坐标间转换，并查询现有 `safe_mask.png` 和 `esdf.npy`。

canonical JSON 摘要的首版算法固定为：解析 JSON 后，以 UTF-8、递归 key 排序、无多余空白、`ensure_ascii=false` 重新序列化，再计算 SHA-256。算法名称、版本和输入 manifest 摘要写入输出 `optimization_manifest.json`，不能依赖普通 `json.dump()` 的原始格式。

每个现有 `episode_XXXXXX.npz` 新增：

- `astar_path_pixels`：`int32`，shape `[M, 2]`，列顺序严格为 `[row, col]`。

现有 `points` 继续作为经过安全平滑和重采样的 `J_ref` 参考曲线，现有 `yaw` 提供平滑路径切线参考和默认边界 yaw。优化器将 `astar_path_pixels` 转成世界坐标后重新积分得到 `S_A*`，并与 manifest 已有的 `raw_path_length_m` 在配置容差内交叉校验。CLI 覆盖 yaw 时不修改输入 episode，只在输出 manifest 记录 effective boundary yaw。

若 `astar_path_pixels`、完整 `MapTransform` 或现有地图文件缺失，优化脚本直接失败；第一阶段不重新运行 A*，也不通过起终点或 `raw_path_length_m` 猜测缺失的完整路径。

对 `generate_sage3d_trajectories.py` 的修改验收标准：

- A*、路径平滑、episode 接受/拒绝和 RNG 顺序不变；
- 已有 episode NPZ 数组值不变，只新增 `astar_path_pixels`；
- 已有 manifest 字段值和语义不变，只新增本节列出的 MapTransform 和 mesh 摘要字段；
- `map/safe_mask.png` 和 `map/esdf.npy` 的值与语义不变，不生成重复副本；
- 原渲染、打包和 LeRobot 流程在出现未知 NPZ key 和 manifest 字段时行为及输出不变；
- 保存 A* 点列不得改变 episode 编号或随机数消费顺序。

### 8.3 独立优化脚本接口

建议的新脚本接口：

```bash
python optimize_sage3d_trajectories.py \
  --input-trajectory-dir /path/to/generated/trajectories \
  --output-dir /path/to/optimized/trajectories \
  --config configs/trajectory_optimization_v1.json \
  [--collision-usd /portable/path/to/collision.usd]
```

脚本必须拒绝：

- 输入和输出解析为同一路径；
- manifest 和 episode 编号不一致；
- episode 缺少 `astar_path_pixels`，或 manifest 缺少完整 `MapTransform`；
- 缺少优化所需的地图或路径数据。

第一阶段输出使用独立研究格式：

```text
<optimized-trajectory-dir>/
├── optimization_manifest.json
├── episode_000000.npz          # time、pose、速度、加速度、jerk
├── episode_000001.npz
└── mecanum_diagnostics/
    ├── summary.json
    ├── episodes.csv
    └── violations.csv
```

它不伪装成现有 SAGE3D/LeRobot 生产格式。生产格式适配必须在第一阶段验收后单独决策。

首版 `optimization_manifest.json` schema version 固定为 `vln_data_prep.trajectory_optimization_output.v1`。它保存：

- 输入 manifest 和 config 的 SHA-256；
- 完整 effective config；
- 软件版本、Git commit、Python 和求解器版本；
- batch 状态和各失败原因汇总；
- 每个 episode 的 `status`、`executable`、全部 `violations`、`primary_failure_reason`；
- 求解器元数据、目标函数分项、验证峰值和诊断摘要。

只为通过第一阶段独立验证的 episode 写优化 NPZ。首版 NPZ 使用 `np.savez_compressed`，字段固定为：

| 字段 | dtype | shape | 语义 |
| --- | --- | --- | --- |
| `time_s` | `float64` | `[K]` | `0, dt, ..., T_output` |
| `pose_world` | `float64` | `[K, 3]` | `[x_m, y_m, yaw_wrapped_rad]` |
| `yaw_unwrapped_rad` | `float64` | `[K]` | 连续 yaw |
| `velocity_world_mps` | `float64` | `[K, 2]` | `[vx, vy]` |
| `yaw_rate_radps` | `float64` | `[K]` | yaw rate |
| `acceleration_world_mps2` | `float64` | `[K, 2]` | `[ax, ay]` |
| `yaw_acceleration_radps2` | `float64` | `[K]` | yaw acceleration |
| `jerk_world_mps3` | `float64` | `[K, 2]` | `[jx, jy]` |
| `yaw_jerk_radps3` | `float64` | `[K]` | yaw jerk |

`solver_metadata` 只保存在 manifest，避免 NPZ 和 manifest 出现两份不一致权威。NPZ 同时保存 wrapped/unwrapped yaw；`pose_world` 的 yaw 明确是 wrapped 版本。

结构性输入错误立即终止整个命令。单 episode 初始化、求解或验证失败记录后继续处理其他 episode，不为失败 episode 创建新的 NPZ。第一阶段验证通过但麦克纳姆诊断超限的 NPZ 可以保留用于研究，但 manifest 必须设置 `executable=false`，并写入 `WHEEL_SPEED_FAILED` 或 `WHEEL_ACCEL_FAILED`；只有所有已启用的可执行性验证通过时才能设置 `executable=true`。

输出目录允许不存在、为空或包含已有文件。程序直接写入目标目录，并覆盖本次成功 episode 的同名 NPZ 和可视化图片；不扫描、不删除本次未计算的旧文件。失败 episode 不写新 NPZ，即使目录中存在旧的同名文件，当前 `candidate_metadata.json` 仍以 `success=false` 和 `npz_filename=null` 明确表示该文件不属于本次有效结果。首版不使用 staging 或原子发布，也不支持 resume、跳过已有 episode、合并历史 metadata 或隐式目录清理；中断可能留下不完整文件。

当前 6.4 candidate CLI 中，`--episode-index` 可选：指定时只计算该 episode，省略时按输入 manifest 的 `episode_count` 计算 `0..episode_count-1`。两种模式统一写 `candidate_metadata.json`；顶层保存 scene、effective config、`requested/succeeded/failed` 汇总和 `episodes` 数组。数组包含本次完成计算的所有 episode，包括数值求解失败项；只在整个调用结束时直接覆盖一次 metadata。任一数值失败不会阻止后续 episode，但最终命令返回非零状态。metadata 是判定本次有效 candidate 的唯一权威，不根据目录中遗留的 NPZ 推断成功或恢复进度。

### 8.4 新功能的内部代码边界

`optimize_sage3d_trajectories.py` 只负责参数解析、读取输入、逐 episode 调用和写出结果。建议为其建立小型 `trajectory_optimization` 包，把内部高度耦合的数值计算放在一个可独立验证的边界内：

- B-spline 求值、导数和时间缩放可以使用合成输入做单元测试；
- 优化目标、硬约束和求解失败可以独立于场景加载与产物写入进行调试；
- 独立验证器不会和求解器共享同一段判断逻辑；
- 麦克纳姆运动学诊断可以读取同一个内部时序轨迹对象；
- 避免把所有新公式和求解器回调继续写入当前生成脚本。

建议的最小文件边界为：

```text
trajectory_optimization/
├── __init__.py
├── spline.py          # B-spline、导数、时间缩放
├── optimizer.py       # 输入、初始化、目标、约束和求解
├── validation.py      # 独立高密度验证
└── mecanum.py         # 只读运动学诊断

tests/isaac/trajectory_optimization/
├── test_spline.py
├── test_optimizer_synthetic.py
├── test_validation.py
└── test_mecanum.py
```

这四个文件都只服务于独立优化脚本，不移动或重新包装原生成器中的已有函数。暂不创建更多单用途模块。

轨迹优化代码使用当前 SAGE3D 轨迹生成所指定的 Isaac Python 环境运行，可以直接使用其中已有的 NumPy、SciPy、trimesh 和 pxr 依赖。测试命令使用：

```bash
export SAGE3D_ISAAC_PYTHON=/ssd4/envs/isaac_sim_py311/bin/python
PYTHONPATH=. "$SAGE3D_ISAAC_PYTHON" -m pytest \
  tests/isaac/trajectory_optimization
```

## 9. 麦克纳姆运动学诊断器

### 9.1 目的

第一阶段不把轮速约束直接并入 NLP，但必须回答：

> 仅使用世界平移、yaw 和导数限制后，产生的轨迹有多少会导致单轮转速或轮加速度超限？

诊断器只读取已经通过第一阶段验证的轨迹，不修改轨迹，也不把违规静默裁剪到上限。

### 9.2 实现前必须确认的底盘约定

必须由实际机器人或仿真模型确认：

- 轮半径 `r`；
- 底盘中心到轮轴的半长 `l_x` 和半宽 `l_y`；
- 四个轮子的编号；
- 四个轮子的物理位置；
- X 型或 O 型滚子布局；
- 每个轮子的滚子方向；
- 正向电机转动的符号；
- body 坐标轴和正 yaw 约定；
- 最大持续轮速和允许的短时峰值轮速；
- 轮速单位；
- 轮加速度限值的物理来源；
- 轮速和轮加速度各自的预留余量；
- 峰值轮速允许持续的最长时间。

> **【需要用户输入】** 请提供真实机器人、URDF/USD、控制器代码或权威规格中的至少一种，并明确上述参数。若不同来源冲突，还需要指定哪个来源具有最终权威。在这些输入齐备前，只实现参数化矩阵和合成测试，诊断结果必须标记为 `physical_parameters_unverified`，不得设置 `executable=true`。

在这些约定确认前，不允许把某个网上常见的正负号矩阵当成真实底盘结论。

### 9.3 运动学计算

世界系速度转换到底盘系：

$$
v^B=R(\theta)^Tv^W.
$$

定义实际底盘约定对应的轮速矩阵 `M`：

$$
\omega_w=
\frac{1}{r}
M
\begin{bmatrix}
v_x^B\\v_y^B\\\dot\theta
\end{bmatrix}.
$$

底盘系平移加速度必须包含旋转坐标系耦合项：

$$
\dot v^B=R(\theta)^Ta^W-\dot\theta Jv^B.
$$

轮加速度：

$$
\alpha_w=
\frac{1}{r}
M
\begin{bmatrix}
\dot v_x^B\\\dot v_y^B\\\ddot\theta
\end{bmatrix}.
$$

运动学单元测试至少覆盖：

- 纯前进；
- 纯横移；
- 纯原地旋转；
- 前进与旋转叠加；
- 世界坐标旋转 90° 后的等价运动；
- 数值微分轮速与解析轮加速度的一致性。

### 9.4 诊断指标

对轮 `j` 定义归一化利用率：

$$
u_{\omega,j}(t)=
\frac{|\omega_{w,j}(t)|}{\omega_{w,max}},
\qquad
u_{\alpha,j}(t)=
\frac{|\alpha_{w,j}(t)|}{\alpha_{w,max}}.
$$

每条轨迹记录：

- 四轮峰值和峰值发生时间；
- 轮速、轮加速度最大利用率；
- 超限采样点比例；
- 超限持续时间和最长连续超限段；
- 最常成为瓶颈的轮子；
- 峰值对应的世界速度、body twist、yaw rate 和曲率；
- 若只统一延长时间，同时修复轮速和轮加速度所需的最小倍率；
- 所需新总时间是否超过 `T_max`。

固定曲线下，轮速随 `1/T` 缩放，轮加速度随 `1/T^2` 缩放。轮速和轮加速度使用独立余量 `eta_omega`、`eta_alpha`，所需保守 retiming 倍率为：

$$
\rho_{retime}
=
\max\left(
1,
\frac{\max_{j,t}u_{\omega,j}(t)}{\eta_\omega},
\sqrt{\frac{\max_{j,t}u_{\alpha,j}(t)}{\eta_\alpha}}
\right).
$$

“理论 retiming 后可修复”表示轮速和轮加速度必须同时通过，且 `rho_retime * T_output <= T_max`。若轮加速度参数尚未得到权威确认，报告可以计算候选值，但不能给出“可执行”结论。

诊断器应同时报告：

- 原始优化结果；
- 允许统一 retiming 后的理论结果；
- 因 `T_max` 或数据时长策略而无法 retiming 的结果。

诊断统计规则固定为：

- 使用独立验证器的高密度时间网格；
- 利用率跨越阈值的起止时间在相邻样本间做线性插值；
- 超限持续时间按插值后的连续时间区间计算，不按样本数乘 `dt`；
- 平移速度低于 `curvature_speed_epsilon_mps` 时曲率记为 `null`，不输出无穷大；
- 多轮并列峰值时记录全部并列轮，单值统计按配置中的轮序取最小索引；
- 分别计算“只保留平移 twist”“只保留 yaw rate”和“完整 twist”的轮速；若前两者单独均未超限而完整 twist 超限，则标记为耦合超限；
- 任意时刻超过峰值轮速立即违规；处于持续上限和峰值上限之间时，只有连续时长超过允许峰值持续时间才违规。

### 9.5 诊断输出

最小输出：

```text
mecanum_diagnostics/
├── summary.json
├── episodes.csv
└── violations.csv
```

`summary.json` 记录底盘参数、限制、余量、数据集摘要和触发结论。`episodes.csv` 每条轨迹一行，`violations.csv` 只记录超限区间，避免保存所有正常采样点造成无意义膨胀。

### 9.6 安全规则与第二阶段触发门槛

安全规则：

> 无论总体超限率多低，任何仍然超过权威轮速或轮加速度限制的轨迹都不能作为可执行专家轨迹发布。

是否进入第二阶段则依据固定基准集。建议当任一条件成立时触发第二阶段：

1. 超过 5% 的第一阶段合格轨迹出现至少一次轮速超限；
2. 按第 9.4 节同时考虑轮速和轮加速度的统一 retiming 后，仍有超过 1% 的轨迹因 `T_max` 或时长策略无法修复；
3. 统一 retiming 使数据集 p50 总时间增加超过 10%；
4. 因轮速验证而拒绝轨迹后，端到端通过率降到 90% 以下；
5. 超限主要来自平移、横移和旋转耦合，单纯延长时间虽可修复但严重损害时间目标。

这些比例是第一轮工程门槛。真实底盘限值、数据规模或生成吞吐目标变化时，应通过一次计划修订显式调整，不能在实现中静默改变。

## 10. 第二阶段：轮速约束进入主优化器

### 10.1 触发条件

只有第 9.6 节的固定诊断基准触发门槛后，才进入本阶段。若绝大多数超限都能用小幅统一 retiming 修复，可以先采用 `T` 初始化下界和最终 retiming，而不立即增加完整非线性轮速约束。

### 10.2 第二阶段目标

第二阶段要证明：

> 在不降低碰撞安全和边界状态正确性的前提下，把真实麦克纳姆轮速限制加入优化器，显著降低轮速拒绝率，并优于对所有轨迹统一放慢的策略。

### 10.3 工作包一：冻结并验证实际轮系约定

- 将已确认的轮编号、符号、几何尺寸和限制写入显式配置；
- 用真实仿真模型或底盘控制接口做正向运动对照；
- 对纯前进、横移和旋转逐项确认轮速符号；
- 禁止在运行时根据结果猜测或自动翻转符号。

验证：解析轮速必须与仿真器或权威底盘模型在约定容差内一致。

### 10.4 工作包二：改善初始总时间

对初始 B-spline 计算轮速和轮加速度峰值。定义 `T_wheel=max(T_wheel_speed,T_wheel_accel)`；若轮加速度参数尚未得到权威确认，则只使用 `T_wheel_speed`，但不得据此标记为物理可执行。初始时间加入：

$$
T_{init}
=
\gamma\max(T_v,T_a,T_j,T_{wheel},T_{min}).
$$

这一步先减少初值的大规模轮速违反，使主 NLP 更容易进入可行域。

### 10.5 工作包三：加入轮速硬约束

在每个非零 knot span 放置轮速配点：

$$
|\omega_{w,j}(u_k)|
\le
\eta_\omega\omega_{w,max},
\qquad j=1,\ldots,4.
$$

约束依赖：

- 平移 B-spline 一阶导数；
- yaw；
- yaw rate；
- 总时间 `T`；
- 轮系矩阵和轮半径。

因此这是非线性约束。第一版不使用有限 penalty 假装硬约束。

优化完成后仍必须使用远高于优化配点密度的独立轮速验证。若峰值落在配点之间，则只在违规区间增加配点并 warm start 重求解。

### 10.6 工作包四：轮速失败恢复

恢复顺序固定为：

```text
局部增加轮速配点
    ↓
在 T_max 内增加初始/下界时间并重新优化
    ↓
允许控制点和 yaw 联合调整
    ↓
拒绝该轨迹
```

不得在优化后逐采样点裁剪轮速，因为这会破坏位置、时间和轮速之间的一致性。

### 10.7 工作包五：轮加速度的条件升级

轮加速度不是轮速约束的自动附属项，也不等价于电机转矩限制。第二阶段先保留轮加速度诊断。

仅当以下任一条件成立时，将轮加速度作为第二阶段的后续子阶段硬约束：

- 超过 5% 的轮速合格轨迹仍有轮加速度超限；
- 超过 1% 的轨迹无法通过 `T <= T_max` 的统一 retiming 修复；
- 已确认轮加速度限值来自可信硬件或控制器约束，而非暂定舒适性数字。

加入后约束为：

$$
|\alpha_{w,j}(u_k)|
\le
\eta_{\alpha}\alpha_{w,max}.
$$

由于它同时耦合世界加速度、yaw、yaw rate 和 yaw acceleration，必须单独记录求解成功率变化，不能与轮速约束一次性混入而失去归因能力。

### 10.8 第二阶段对照实验

在完全相同的固定路径集上比较：

1. 第一阶段原始结果；
2. 第一阶段结果统一 retiming；
3. 加入轮速硬约束的联合优化结果。

比较指标：

- 轮速和轮加速度超限率；
- 端到端通过率；
- 总时间 p50/p95；
- 路径长度；
- 平移和 yaw jerk；
- clearance；
- 求解时间和失败原因；
- 相对统一 retiming 节省的总时间。

### 10.9 第二阶段验收门槛

第二阶段完成需要：

1. 所有被接受轨迹的轮速独立验证零超限。
2. 碰撞、边界状态和第一阶段导数限制没有回归。
3. 因轮速导致的拒绝率不高于 1%。
4. 固定基准的端到端通过率恢复到至少 90%。
5. 相比统一 retiming，联合优化在固定基准上有可测量的时间或通过率收益。
6. 轮系约定和限制被记录到诊断产物，结果可复现。

如果第 5 条不成立，说明完整轮速 NLP 没有证明其复杂度价值，应保留更简单的统一 retiming 方案。

## 11. 后续阶段候选，但不在当前计划内

第一、二阶段稳定后，才按证据选择后续优先级：

- 正式凸安全走廊与 Bézier extraction；
- 少量分组 span 时长变量；
- 非零边界速度和状态可行性采样；
- 多条 A* 候选路径优化后比较；
- 相机朝向未来运动方向的显式目标或约束；
- 生产 NPZ/manifest 的时间、速度字段迁移；
- 渲染 FPS、控制周期和 LeRobot 数据语义统一；
- 仿真跟踪误差和真实底盘执行验证；
- 必要时评估 CasADi/IPOPT。

非零初速度的“紧急避障”应作为单独需求处理。任意采样的速度状态可能没有无碰撞制动轨迹，因此需要可达性筛选、制动距离检查或 kinodynamic 搜索，不能仅通过放宽第一阶段边界条件实现。

## 12. 阶段决策摘要

```text
第一阶段：静止到静止的 B-spline 联合优化原型
    ↓ 独立验证通过
运动学诊断：轮速/轮加速度只读统计 + retiming 分析
    ├── 超限少且小幅 retiming 可修复
    │       └── 保持简单方案，不进入完整轮速 NLP
    └── 达到量化触发门槛
            ↓
第二阶段：轮速进入主优化器
            ↓
      轮加速度继续诊断
            ├── 未触发门槛：保持验证/拒绝
            └── 触发门槛且限值可信：加入轮加速度硬约束
```

每次阶段升级都必须由固定基准报告触发，而不是因为后续功能在理论上可能有用。
