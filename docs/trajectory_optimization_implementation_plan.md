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
3. 第一阶段使用独立输入、输出目录，不覆盖原始轨迹；原生成器只允许增加 sidecar 中间产物。
4. 碰撞、安全边界和导数上限属于验收条件，不用大权重近似替代。
5. 先用当前已安装的 NumPy、SciPy 和 OSQP；没有基准证据前不引入 CasADi/IPOPT。
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
- 不保存原始 A* 完整点列、`safe` 栅格、`clearance_m` 数组和地图坐标变换的可独立读取副本。

现有 episode NPZ 中的 `points` 可以作为由 A* 派生的几何参考路径，但仅靠这些点不足以可靠建立碰撞约束。新的优化器不是现有 `smooth_path()` 的局部替换，而是读取原生成结果和额外规划上下文的独立后处理器。

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
- 改变现有 episode NPZ、manifest、渲染或 LeRobot 字段的语义；允许新增不会被现有消费者误读的 sidecar 文件。

## 5. 第一阶段目标

第一阶段要证明：

> 给定原生成器的轨迹目录和只读规划上下文，独立优化脚本能够稳定产生一条具有真实时间含义的静止到静止五次 B-spline 轨迹；所有被接受的轨迹都通过独立碰撞和导数限制验证。

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

必须在实现前确认：

- 控制周期 `dt`；
- `T_min`、`T_max`；
- 平移速度、加速度和 jerk 限值；
- yaw rate、yaw acceleration 和 yaw jerk 限值；
- 数值验收容差和安全余量；
- 限值来自硬件、仿真器还是数据质量要求。

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

构造分段线性参考曲线 `r(s)`，并按路径长度动态确定控制点数量。控制点数量必须设置最小值和最大值，避免短路径表达过度和长路径表达不足。

平移控制点初始化使用一个小型约束最小二乘或 QP：

$$
\min_P
\sum_i\|P_i-\widetilde P_i\|^2
+
\lambda_{init}
\sum_i\|P_{i+1}-2P_i+P_{i-1}\|^2.
$$

其中 `P_tilde` 来自按弧长均匀采样的 A* 参考路径。初始化平滑项只用于得到良好初值，不代表真实物理加速度。

yaw 初始化采用：

1. 起终 yaw 最短角度 unwrap；
2. 路径切线作为中间软参考；
3. 起终 yaw 和 yaw rate 边界精确满足。

初始时间根据导数控制点估计：

$$
T_{init}=\gamma\max(T_v,T_a,T_j,T_{min}),
\qquad \gamma>1.
$$

验证：初始曲线必须通过端点、满足零边界速度，并且在进入主优化器前生成完整的碰撞与导数诊断报告。

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

第一阶段硬约束：

- 起终位置和 yaw；
- 起终世界速度和 yaw rate 为零；
- `T_min <= T <= T_max`；
- 平移速度、加速度和 jerk 上限；
- yaw rate、yaw acceleration 和 yaw jerk 上限；
- 优化配点处的地图 clearance 约束；
- 控制点相对初始解的保守 trust region。

碰撞约束使用 `clearance_m` 的双线性插值，但最终是否安全仍由独立 `safe_mask` 和高密度碰撞验证决定。第一阶段不宣称有限配点给出了连续碰撞证明。

求解器顺序：

1. 使用 SciPy SLSQP 快速验证变量、目标和约束方向；
2. 若基准显示 SLSQP 不稳定，再比较 `trust-constr`；
3. 只有在固定基准证明现有求解器是主要瓶颈后，才单独评估 CasADi/IPOPT。

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
NUMERICAL_FAILURE
```

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
6. 固定输入重复运行在约定数值容差内一致。
7. 优化后的归一化总目标不高于其初始解。
8. 生成完整的麦克纳姆运动学诊断报告。
9. 现有 episode NPZ 和 manifest 内容保持原有语义，渲染和打包代码保持不变；新增 sidecar 不被现有消费者扫描为 episode。

90% 是研究原型门槛，不是生产发布门槛。生产集成前需要单独制定更高的成功率和吞吐量目标。

## 8. 独立脚本、输入 sidecar 与代码组织

### 8.1 两步运行边界

轨迹生成和轨迹优化是两个独立进程：

```text
generate_sage3d_trajectories.py
    ├── 保持当前 A*、几何平滑和原始产物输出
    └── 额外保存只读 optimization_inputs sidecar

optimize_sage3d_trajectories.py
    ├── 读取原始 trajectory directory
    ├── 读取 optimization_inputs sidecar
    ├── 执行 B-spline 优化、验证和运动学诊断
    └── 写入另一个 optimized trajectory directory
```

`optimize_sage3d_trajectories.py` 不导入或调用 `generate_sage3d_trajectories.py`，不重新运行 A*，也不原地修改输入目录。若优化失败，原始生成结果保持可用。

### 8.2 原生成器需要增加的最小 sidecar

当前 episode 的 `points` 已经提供由 A* 派生的平滑参考路径。为了让独立脚本准确复现规划时的安全语义，原生成器只额外保存以下内容：

```text
<trajectory-dir>/
├── trajectory_manifest.json             # 现有文件，不改变原有字段语义
├── episode_000000.npz                    # 现有文件，不增加优化专用字段
├── episode_000001.npz
└── optimization_inputs/
    ├── context.json
    ├── map.npz
    ├── episode_000000.npz
    └── episode_000001.npz
```

`optimization_inputs/context.json` 至少记录：

- sidecar schema version；
- scene ID 和生成 seed；
- `MapTransform` 的 `height`、`width`、`scale`、`lower_x`、`lower_y`；
- `required_path_clearance_m`；
- `safe_mask` 已包含的机器人和相机安全语义；
- 原始 manifest 的内容摘要，用于防止目录混配。

`optimization_inputs/map.npz` 保存：

- `safe_mask`，布尔数组；
- `clearance_m`，浮点数组。

每个 `optimization_inputs/episode_XXXXXX.npz` 保存该 episode 的原始 `astar_path_pixels`。优化脚本利用 context 中的 `MapTransform` 转成世界坐标。现有 episode NPZ 中的 `points` 继续作为经过安全平滑和重采样的参考曲线，两者不重复保存。

sidecar 子目录不能命名为根目录下的 `episode_*.npz`，避免现有渲染或打包逻辑把它识别为新的 episode。若 sidecar 缺失或与 manifest 摘要不匹配，优化脚本应明确失败；第一阶段不通过重新推断地图来静默兼容旧产物。

对 `generate_sage3d_trajectories.py` 的修改验收标准：

- A*、路径平滑、episode 接受/拒绝和 RNG 顺序不变；
- 已有 episode NPZ 数组值不变；
- 已有 manifest 字段值和语义不变；
- 只新增 `optimization_inputs/`；
- sidecar 可被独立加载并与原 episode 一一对应。

### 8.3 独立优化脚本接口

建议的新脚本接口：

```bash
python optimize_sage3d_trajectories.py \
  --input-trajectory-dir /path/to/generated/trajectories \
  --output-dir /path/to/optimized/trajectories \
  --control-dt 0.1
```

脚本必须拒绝：

- 输入和输出解析为同一路径；
- 输出目录中已有不相关内容；
- manifest、sidecar 和 episode 编号不一致；
- sidecar schema version 不受支持；
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

轨迹优化代码使用当前 SAGE3D 轨迹生成所指定的 Isaac Python 环境运行，可以直接使用其中已有的 NumPy、SciPy、OSQP、trimesh 和 pxr 依赖。测试命令使用：

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
- 每个轮子的滚子方向；
- 正向电机转动的符号；
- 最大持续轮速和允许的短时峰值轮速；
- 轮加速度限值的物理来源；
- 是否需要预留控制余量，例如只使用额定上限的 90%。

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
- 若只统一延长时间，修复轮速所需的最小倍率；
- 所需新总时间是否超过 `T_max`。

固定曲线下，轮速随 `1/T` 缩放。若预留利用率上限为 `eta < 1`，所需保守 retiming 倍率为：

$$
\rho_{retime}
=
\max\left(1,\frac{\max_{j,t}u_{\omega,j}(t)}{\eta}\right).
$$

诊断器应同时报告：

- 原始优化结果；
- 允许统一 retiming 后的理论结果；
- 因 `T_max` 或数据时长策略而无法 retiming 的结果。

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

> 无论总体超限率多低，任何仍然轮速超限的轨迹都不能作为可执行专家轨迹发布。

是否进入第二阶段则依据固定基准集。建议当任一条件成立时触发第二阶段：

1. 超过 5% 的第一阶段合格轨迹出现至少一次轮速超限；
2. 允许统一 retiming 后，仍有超过 1% 的轨迹因 `T_max` 或时长策略无法修复；
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

对初始 B-spline 计算轮速峰值，并加入：

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
\eta\omega_{w,max},
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
