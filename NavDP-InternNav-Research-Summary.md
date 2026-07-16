# NavDP & InternNav 调研总结

> 本文档梳理了关于 InternRobotics 发布的 NavDP 与 InternNav 两个仓库、InternData-N1 数据集、Scene-N1 场景资产的完整调研结论。所有信息均来源于本地仓库代码、HF 数据卡、NavDP 论文 (arXiv:2505.08712 v3)、InternNav 官方文档以及 InternRobotics GitHub 组织的 80 个仓库的交叉验证。

---

## 1. 两个仓库的角色分工

### 1.1 NavDP 仓库 (`/ssd4/github-knowledge-base/NavDP`)

**定位：推理 + 评测，不提供训练代码也不提供数据生成代码。**

| 关注点 | 是否提供 | 位置 |
|---|---|---|
| 模型结构代码 | ✅ | `baselines/navdp/policy_network.py:9` (`NavDP_Policy`)、`policy_backbone.py` (RGBD/PixelGoal/ImageGoal backbones)、`policy_agent.py`、`navdp_server.py` |
| 训练代码 | ❌ | 整个仓库只有 `eval_*_wheeled.py` / `teleop_*_wheeled.py` / `navdp_server.py`。`grep optimizer.step / loss.backward` 仅命中 `baselines/navdp/depth_anything/metric_depth/train.py`（DepthAnything 训练，非 NavDP） |
| 数据处理代码 | ❌ | 仓库定位是 "benchmark + baseline server"，README 直接指向 HuggingFace `InternData-N1` 数据集 |

- 该仓库等于"权重 + 推理服务 + IsaacSim 评测"，配置上要求 IsaacSim 4.2.0 + IsaacLab 1.2.0
- README 第 33 行直接指向 HF `InternData-N1`，没有数据生成/转换脚本
- 评测入口：`eval_pointgoal_wheeled.py` / `eval_imagegoal_wheeled.py` / `eval_nogoal_wheeled.py` / `eval_startgoal_wheeled.py`，全部通过 HTTP 调用 `navdp_server.py` 提供的 API

### 1.2 InternNav 仓库 (`/ssd4/github-knowledge-base/InternNav`)

**定位：all-in-one 训练栈，提供完整的 NavDP 训练代码与数据加载器。**

| 关注点                     | 是否提供   | 位置                                                                                                                                                                                                                                                                                                                  |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模型结构代码（更完整）             | ✅      | `internnav/model/basemodel/navdp/navdp_policy.py:34` (`NavDPNet`，HF `PreTrainedModel` 包装)；`internnav/model/encoder/navdp_backbone.py` (RGBDBackbone / ImageGoalBackbone / PixelGoalBackbone / 位置编码)；另有 `internnav/model/basemodel/internvla_n1/navdp.py:16` 给出 `NavDP_Policy_DPT_CriticSum_DAT` 用于 InternVLA-N1 双系统 |
| 训练代码                    | ✅      | `internnav/trainer/navdp_trainer.py:11` (`NavDPTrainer.compute_loss`)；入口 `scripts/train/base_train/train.py:170,219`；启动脚本 `scripts/train/base_train/start_train.sh`（`--model navdp` 走 8-GPU `torchrun`）；超参 `scripts/train/base_train/configs/navdp.py`                                                              |
| 数据加载（训练时用）              | ✅      | `internnav/dataset/navdp_lerobot_dataset.py:34` (`NavDP_Base_Datset`, `navdp_collate_fn`)，从 LeRobot 格式读 `videos/observation.images.rgb`、`observation.images.depth`、`meta/episodes_stats.jsonl`、`meta/pointcloud.ply`；旧版 `internnav/dataset/navdp_dataset.py` 也保留                                                    |
| 从原始数据 → NavDP 训练样本的转换脚本 | ⚠️ 仅部分 | `scripts/dataset_converters/vlnce2lerobot.py` 是 VLN-CE → LeRobot 的转换，并非专门给 NavDP 用。NavDP 训练消费的 `data/datasets/InternData-N1/vln_n1/traj_data` + `navdp_dataset_lerobot.json`（见 `configs/navdp.py:51-52`）需要从 HuggingFace `InternData-N1` 直接下载，仓库内 grep 不到生成它的脚本                                                      |

#### 关键训练入口

**启动训练命令：**
```bash
bash scripts/train/base_train/start_train.sh --name navdp_train --model navdp
```
该脚本会自动设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` + `NUM_GPUS=8`，并通过 `torchrun` 启动 `scripts/train/base_train/train.py`。

**关键超参 (`scripts/train/base_train/configs/navdp.py`)：**
```python
navdp_exp_cfg = ExpCfg(
    name='navdp_train',
    model_name='navdp',
    il=IlCfg(
        epochs=1000,
        batch_size=32,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        dataset_navdp='data/datasets/navdp_dataset_lerobot.json',
        root_dir='data/datasets/InternData-N1/vln_n1/traj_data',
        image_size=224,
        scene_scale=1.0,
        memory_size=8,
        predict_size=24,
        pixel_channel=4,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        dropout=0.1,
        filter_failure=FilterFailure(use=True, min_rgb_nums=15),
        loss=Loss(alpha=0.0001, dist_scale=1),
    ),
    model=navdp_cfg,
)
```

---

## 2. NavDP 训练用的数据来源

### 2.1 训练数据集：仅 `vln_n1` 一个子集

**NavDP 训练 = `InternData-N1/vln_n1/traj_data/` 这一份。不用 `vln_ce`，也不用 `vln_pe`。**

三层证据：

**(1) NavDP 论文本身** — 从头到尾没出现 "VLN-CE / VLN-PE / R2R / RxR / instruction" 这些词。NavDP 任务是 PointGoal / ImageGoal / NoGoal，goal 用 2D 相对坐标 `(x_g, y_g)` 或图片表达，no-goal 直接喂全零向量，不接受自然语言指令。

**(2) InternNav 训练配置 (`scripts/train/base_train/configs/navdp.py:51-52`)：**
```python
dataset_navdp='data/datasets/navdp_dataset_lerobot.json',
root_dir='data/datasets/InternData-N1/vln_n1/traj_data',
```
明确指向 `vln_n1/traj_data/`，没有 `vln_ce` / `vln_pe` 入口。

**(3) HF dataset card 子集分工：**
| 子集 | 数量 | 用途定位 |
|---|---|---|
| **vln_n1** | 660K+ instructions / 210K+ videos | "Synthetic Data for InternVLA-N1" — 这就是 NavDP 使用的轨迹源 |
| vln_ce | 170K+ episodes | VLN-CE benchmark (Habitat 离散动作) |
| vln_pe | 8K+ episodes | VLN-PE benchmark (Isaac Sim 物理仿真) |

`vln_n1` 的 210K 视频数与论文 200K 轨迹数完全对得上 → 它就是论文 DataEngine 产出的那批数据的开源版本，只是顺带带了 LLM rephrase 的语言指令（用于上层 InternVLA-N1 System-2 reasoning），NavDP 训练时这些文本字段被忽略。

### 2.2 vln_ce / vln_pe 的用途

**vln_ce / vln_pe 不是 evaluation-only，而是各自支持独立 benchmark 的"训练 + 评测"两条管线**，主要给 InternNav 工具箱里其他 baseline（CMA、Seq2Seq、RDP 等 VLN 模型）用，跟 NavDP 无关。

| 子集 | 任务范式 | 仿真器 | 用途 |
|---|---|---|---|
| **vln_ce** | VLN-CE：自然语言指令 + 离散动作 (Forward/TurnLeft/...) | Habitat | 训练 / 评测 R2R、RxR 等经典 VLN 模型；InternNav 里的 CMA、Seq2Seq、RDP baselines 都吃它 |
| **vln_pe** | VLN-PE：语言指令 VLN，但放在物理仿真里跑 (人形/四足/轮式) | Isaac Sim | InternNav 提出的 "physically-realistic" benchmark (arXiv:2507.13019)，既给训练数据也给评测数据 |
| **vln_n1** | 纯几何 PointGoal/ImageGoal/NoGoal 轨迹 (200K 条 / 3000+ 场景) | BlenderProc 渲染 | NavDP 训练用；上面附带的 LLM rephrase 指令给 InternVLA-N1 System-2 |

证据：
- HF dataset card 把 vln_ce / vln_pe 都明确划成 "benchmark" (各自带 `train / val_seen / val_unseen` 划分)
- InternNav 仓库 `internnav/dataset/` 下分别有 `cma_lerobot_dataset.py` / `rdp_lerobot_dataset.py` 等独立 dataset 类专门读 vln_ce / vln_pe 数据
- 配置文件 `cma.py / rdp.py / seq2seq.py` 全部消费这两份数据来训练和评测

---

## 3. NavDP 训练数据规模

来自 NavDP 论文 III. DataEngine 节：

- **3000+ 场景**，来自 6 个开源场景数据集：3D-Front、HSSD、HM3D、Replica、Gibson、Matterport3D
- **每场景采样 100 对 (start, goal)**
- **过滤后剩 200K 条轨迹**，覆盖 **100 万米以上**导航距离
- 叠加 **光照 / 视角 / 纹理** 三种 domain randomization
- 表 I 与其他数据集对比 (GoStanford 27 场景 / RECON 9 / SCAND 1 / SACSoN 5 / AMR 54 / **NavDP 3154**)

---

## 4. InternData-N1 数据生成 pipeline 是否开源

### 4.1 生产逻辑（文字描述，公开）

NavDP 论文 III. DataEngine 节明确写出 InternData-N1 中 vln_n1 / NavDP 子集的生成流程：

1. **Robot Model**：圆柱刚体 + 双轮差速，碰撞半径 r_b=0.25m；相机高度在 (0.25m, 1.25m) 内随机，俯仰角 (-30°, 0°) 内随机；FOV 两套 (D435i 69°×42° 和 Zed 2 110°×70°)
2. **Trajectory Generation**：场景 mesh → 0.05m voxel → 估 ESDF；按 h_nav/h_obs 阈值切分可走/障碍区；ESDF 截断到 robot 半径 → 下采样到 0.2m → **A\*** 规划起终点路径 → 局部贪心搜索把 waypoint 推离障碍 → 三次样条插值得到平滑轨迹
3. **Scene Assets & Render**：用 **BlenderProc** 在 3D-Front / HSSD / HM3D / Replica / Gibson / Matterport3D 共 3000+ 场景上渲染照片级 RGB-D；每场景采样 100 对起终点；做 light / view / texture 三类 domain randomization；最终过滤后 200K 条轨迹、1M+ 米
4. **Critic 训练数据**：在 expert 轨迹上做随机旋转扰动得到对比样本，用 ESDF 在线打 critic label

### 4.2 开源代码情况：**生产侧基本没有开源**

| 阶段 | 是否开源 | 证据 |
|---|---|---|
| ESDF 体素化 + A\* + waypoint 优化 + spline | ❌ | NavDP 论文 v3 (2025-12) 仍写 "The dataset will be open-sourced in the near future."；NavDP 仓库 grep 不到 ESDF/A\*/BlenderProc；InternRobotics 80 个 repo 中没有 "NavDataEngine" 类项目 (`InternDataEngine` 是 Manipulation 用的) |
| BlenderProc 渲染脚本 | ❌ | 论文只说"使用 BlenderProc"，没给脚本 |
| InternUtopia 内的轨迹采集 (VLN-PE 用) | ⚠️ Coming Soon | InternNav 文档 *Tutorials → Dataset → "Collect Demonstration Dataset in InternUtopia"* 原文："Support for collecting demos via InternUtopia simulation is coming soon — stay tuned!" |
| VLN-CE 原始数据 → LeRobot | ✅ | `InternNav/scripts/dataset_converters/vlnce2lerobot.py` (650 行)，文档专门介绍用法 |
| LLM 重述指令 (instruction rephrasing) | ❌ | HF dataset card 只在描述里提 "rephrased instructions generated by LLMs"，未公布 prompt/脚本 |
| 数据过滤逻辑 | ❌ | HF card 一句话提及，仓库无对应代码 |
| 训练时的 dataset 读取 | ✅ | `InternNav/internnav/dataset/navdp_lerobot_dataset.py`，但只读已生成数据 |

### 4.3 唯一的辅助数据集开源：Scene-N1

InternRobotics 在 HF 上额外放了 `InternRobotics/Scene-N1` (147GB)，但只覆盖训练场景的 **1/6（仅 Matterport3D 部分）**，其余 3D-Front / HSSD / HM3D / Replica / Gibson 必须按各自原数据集 license 自己下载。

---

## 5. d435i / zed 的含义

`vln_n1/traj_data/` 下的目录后缀是**两套相机内参 (FOV) 配置**的标记，源自 NavDP 论文 III. DataEngine 节：

| 目录后缀 | 相机型号 | HFOV × VFOV |
|---|---|---|
| `*_d435i` | Intel RealSense D435i | 69° × 42° |
| `*_zed` | Stereolabs Zed 2 | 110° × 70° |

**为什么要分两套**：FOV 是"换不掉"的硬件维度，所以两套渲染各跑一遍 → 训练时混合输入，让 NavDP 在不同 FOV 的真机上零样本迁移 (D435i 类轮式机器人 vs Zed 2 类四足/无人机)。

**对换 fisheye 的含义**：轨迹本身 (`action` 列) 和场景 mesh 是相机无关的，d435i / zed 只是 InternRobotics 给的两个采样点。完全可以保留同一组轨迹 + 同一个场景，把内参换成任意鱼眼模型重新跑一次。

### 5.1 vln_n1 完整目录结构

通过 HF API 列出 `vln_n1/traj_data/` 顶层 12 个目录：
```
3dfront_d435i      3dfront_zed
gibson_d435i       gibson_zed
hm3d_d435i         hm3d_zed
hssd_d435i         hssd_zed
matterport3d_d435i matterport3d_zed
replica_d435i      replica_zed
```

每个目录里以 `<scene_id>.tar.gz` 形式存放每个场景的轨迹数据。

---

## 6. InternData-N1 数据 chunk 实际结构

下载的示例 chunk：`/ssd4/github-knowledge-base/00154c06-2ee2-408a-9664-b8fd74742897/` (70MB, 32 episodes, 4228 frames, ~155m 累计轨迹)

### 6.1 目录结构（标准 LeRobot v2.1）

```
00154c06-2ee2-408a-9664-b8fd74742897/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ... episode_000031.parquet
├── videos/chunk-000/
│   ├── observation.images.depth/    (PNG, 480×270 16-bit)
│   ├── observation.images.rgb/
│   ├── observation.video.depth/     (MP4)
│   └── observation.video.rgb/
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── tasks.jsonl
    └── pointcloud.ply               (88,750 彩色点, 2.4MB)
```

### 6.2 parquet schema

每帧一行：
```
index                              int64
observation.camera_intrinsic       float[9]   (3x3 K)
observation.camera_extrinsic       float[16]  (4x4 SE(3) — 相机相对机器人基座，整集恒定)
action                             float[16]  (4x4 SE(3) — 机器人在世界系下的位姿)  ← 这就是轨迹
```

### 6.3 实际数值 (episode_000000, 78 帧)

```
action[0] 平移 = (-4.72, -6.54, 0.357), 偏航 ≈ 10.5°
action[77] 平移 = (-3.95, -3.94, 0.357), 偏航 ≈ -62.3°
z 恒为 0.357m → 平面导航 (机器人基座高度)
32 个 episode 累计 path_len ≈ 155.11 m, 平均每集 ≈ 4.85m
帧间步长 ≈ 0.033 – 0.048 m / frame, 最大 < 0.05m
```

→ 完全对应论文里 0.05m voxel + spline 平滑出来的离散化轨迹。

### 6.4 内参 (D435i 配置)

```
fx=355.81464, fy=351.687, cx=240, cy=135  (480×270)
hfov ≈ 68°,  vfov ≈ 42°
```
**精准匹配 D435i 的 (69°, 42°)** → 这个 chunk 属于 `3dfront_d435i` 子集。

### 6.5 语言信息

`meta/episodes.jsonl` 每个 episode 附带：
- `sub_instruction` / `revised_sub_instruction` (LLM rephrase)
- `sub_indexes` / `sum_indexes` (指令对应的帧区间，如 `[0, 77]`)
- `sum_instruction` (整段汇总指令)

### 6.6 meta 文件无源数据集字段

5 个 meta 文件全部检查过：
| 文件 | 内容 | 提到源数据集？ |
|---|---|---|
| `info.json` | LeRobot v2.1 schema, fps, robot_type="unknown" | ❌ |
| `episodes.jsonl` | episode_index + 指令 + 帧区间 | ❌ |
| `tasks.jsonl` | 指令文本 | ❌ |
| `episodes_stats.jsonl` | episode_index, task_index 范围, image_index 范围 | ❌ |
| `pointcloud.ply` | binary PLY，注释只有 `Created by Open3D` | ❌ |

**唯一可用作源数据集线索的是目录名 UUID 本身**。

### 6.7 Q1: 一个 episode 对应一条轨迹吗？

**是。** 每个 `episode_NNNNNN.parquet` 就是一条独立的轨迹。32 条轨迹都来自同一个场景 (UUID `00154c06-...` 就是 scene id)。证据：
- 各 episode 帧数相互独立 (72–175 帧不等)
- 起点-终点散布在 (-4.7, -6.5) ~ (+4.4, +6.7) 范围内
- 相邻 episode 的 "上一条终点 → 下一条起点" 间距 2.8 – 8.4 m (不连续)
- 每个 episode 在 `episodes.jsonl` 里都对应一段独立的自然语言指令

等价理解：**32 条 (start, goal, instruction, trajectory, frames) 四元组，全部在同一个场景里采样**。

### 6.8 Q2: 能否用轨迹 + 原始 3D 场景，自己重渲染 (如 fisheye)？

**理论上完全可以，但下载的 chunk 缺少最关键的"场景 mesh"。**

重渲染所需 5 样东西：

| 需要的东西 | 在哪里 | 备注 |
|---|---|---|
| 每帧机器人世界位姿 `T_robot^world` | `parquet` 里的 `action` (4×4) | |
| 相机相对机器人基座外参 `T_cam^robot` | `observation.camera_extrinsic` (4×4，整集恒定) | 解码出来是绕 X 轴 -90°、Z=0.357m 的固定标定 |
| 原始相机内参 `K` (用作参考) | `observation.camera_intrinsic` (3×3) | |
| 帧时序索引 | `index` 列 (0…N-1) | 帧率 30 Hz |
| 自由设计 fisheye 内参 / 畸变模型 | 自己定 | KB 鱼眼 / 等距投影 / 等立体角 / 全景都可以 |

相机世界位姿：`T_cam_world[i] = T_robot_world[i] @ T_cam_robot`

**缺的关键一样**：带纹理的原始场景 mesh
- `meta/pointcloud.ply` 只有 88,750 个彩色点 (2.4 MB, 降采版)，**不能直接做光线追踪/光栅化重渲染**
- 文件夹名 `00154c06-...` 是场景 UUID

---

## 7. chunk `00154c06-...` 的源数据集

**通过 UUID 反查命中 3D-FRONT。** Kaggle 上的 3D-FRONT 数据集列表里直接就能看到 `00154c06-2ee2-408a-9664-b8fd74742897.json` 这个文件 — 它是 3D-FRONT house 一级的 scene id。

指令里频繁出现 "black marble wall / patterned floor tiles / wooden round table / black piano / brick wall" 这种装修化描述，也与 3D-FRONT 合成室内风格一致 (HM3D / MP3D / Gibson 是真实扫描，Replica 风格也不同；HSSD 也合成但 id 命名约定不同)。

**结论**：这个 chunk 来自 3D-FRONT，属于 **`vln_n1/traj_data/3dfront_d435i/`** 子集 (3D-FRONT 场景 + D435i FOV 渲染)。

**验证方法**：向阿里天池 3D-FRONT 申请 license 后下载 house JSON 包，查 `3D-FRONT/00154c06-2ee2-408a-9664-b8fd74742897.json` 是否存在。

---

## 8. Scene-N1 仓库结构

### 8.1 顶层目录

```
scene_data/
├── mp3d_pe.tar.gz         28.9 GB   ← 训练+评测用：VLN-PE benchmark 的改进版 MP3D 资产
├── mp3d_n1.tar.gz         25.9 GB   ← 训练用：生成 vln_n1 (NavDP 训练) 轨迹的 MP3D 基础扫描
├── mp3d_ce.tar.gz         16.1 GB   ← 训练+评测用：VLN-CE benchmark 的 MP3D 资产
├── gradio_scene_assets.zip 240 MB   ← 演示/可视化
└── n1_eval_scenes/                  ← 评测专用
    ├── Materials.tar.gz       863 MB
    ├── SkyTexture.tar.gz      103 MB
    ├── cluttered_easy.tar.gz  58 MB    ← NavDP benchmark Cluttered-Easy
    ├── cluttered_hard.tar.gz  61 MB    ← NavDP benchmark Cluttered-Hard
    ├── internscenes_commercial.tar.gz  23.7 GB  ← NavDP benchmark
    └── internscenes_home/
        ├── Materials.tar.gz  19.2 GB
        ├── layout.tar.gz      1.7 GB
        ├── object.tar.gz     29.8 GB
        └── scenes_home.tar.gz 8.2 MB
```

### 8.2 各目录用途

| 目录 | 官方说明 | 训练 / 评测 |
|---|---|---|
| `mp3d_pe/` | "Improved Matterport3D scene assets for VLN-PE benchmark" | 训练 + 评测 (取决于 split) |
| `mp3d_n1/` | "Base Matterport3D scans used for generating VLN-N1 trajectory data" | **训练用** (NavDP 训练数据的源场景) |
| `mp3d_ce/` | "Matterport3D scene assets for VLN-CE benchmark" | 训练 + 评测 |
| `n1_eval_scenes/` | NavDP 仓库 README 第 121 行直接对应到 `assets/scenes/` | **纯评测** (NavDP 论文的 IsaacSim 闭环 benchmark) |
| `gradio_scene_assets.zip` | 演示 | demo |

### 8.3 重要说明

**Scene-N1 ≠ 全部训练场景**。NavDP 训练用的 3000+ 场景来自 6 个数据集，Scene-N1 上只放了其中 **Matterport3D 一份**。原因：
1. **License 限制** — 3D-Front (阿里)、HSSD (FAIR)、Gibson (Stanford)、HM3D (Meta) 都有各自 ToS，InternRobotics 不能二次分发
2. **MP3D 也只是"基础扫描"重新整理** — Scene-N1 README 写 "The original scene datasets can be obtained from Matterport3D"，意思是改进后的版本要配合 MP3D 原始 license 一起拿

### 8.4 "基础扫描"和"重新整理"的含义

InternRobotics 拿原始 MP3D 后重新打包出**三个独立版本**：

| 子目录 | 含义 | 与原始 MP3D 的差别 |
|---|---|---|
| `mp3d_pe/` | "Improved" | **重整理 + 改进**：转成 Isaac Sim 可读 USD 格式，做了网格修复、可碰撞 collider 重建 |
| `mp3d_n1/` | "Base" | **基础版本**：保留 Habitat / BlenderProc 可用的原始 mesh，只做了基本格式整理 |
| `mp3d_ce/` | (VLN-CE) | **VLN-CE 标准版**：与 Habitat-Lab 兼容 (沿用 jacobkrantz/VLN-CE 项目里的 Habitat scene 配置) |

---

## 9. Scene-N1 中 MP3D 部分的实测

解压位置：`/ssd5/datasets/Scene-N1/mp3d_n1/` (41GB)

### 9.1 场景数量

**90 个场景目录** ✅ 与 MP3D 全量一致 (Chang et al. 2017, 90 building scenes)

| 来源 | 场景数 |
|---|---|
| `/ssd5/datasets/Scene-N1/mp3d_n1/` | **90** ✅ |
| `vln_n1/traj_data/matterport3d_d435i/` | 65 |
| `vln_n1/traj_data/matterport3d_zed/` | 66 |
| 两个相机配置并集 | 68 |

→ Scene-N1 的 `mp3d_n1.tar.gz` 给的是**全量 90**，并不是只有 vln_n1 用到的 68 个。NavDP 实际训练用了 68 个，剩下 25 个未参与训练（可能是过滤掉了楼层多/扫描洞多/不可导航区域大的场景）。

**25 个未参与训练的 MP3D 场景：**
```
2t7WUuJeko7  5q7pvUzZiYa  8194nk5LbLH  aayBHfsNo7d  b8cTxDM8gDG
e9zR4mvMWw7  fzynW3qQPVF  GdvgFV5R1Z5  gYvKGZ5eRqb  gZ6f7yhEvPG
HxpKQynjfin  kEZ7cmS4wCh  oLBMNvg9in8  Pm6F8kyY3z2  PX4nDJXEHrG
qoiz87JEwZ2  QUCTc6BB5sX  rPc6DW4iMge  rqfALeAoiTq  UwV83HsGsw3
Uxmj2M2itWa  V2XKFyX4ASd  VVfe2KiqLaN  WYY7iVyf5p8  YVUC4YcDtcY
```

### 9.2 每个场景的内容 (以 17DRP5sb8fy 为例, 199MB)

```
17DRP5sb8fy/
├── matterport_mesh/                                  ← 主 mesh
│   └── bed1a77d92d64f5cbbaaae4feed64ec1/
│       ├── bed1a77d92d64f5cbbaaae4feed64ec1.obj     ← 111K 顶点 / 216K 面, 纹理化 mesh
│       ├── bed1a77d92d64f5cbbaaae4feed64ec1.mtl     ← 材质
│       ├── textures/                                 ← 23 张 .jpg 纹理
│       └── config.yaml                              ← Isaac Sim 转换配置 (提到 isaacsim_*.usd)
├── house_segmentations/                              ← 语义标注
│   ├── *.house                                      ← 房间/物体语义标注
│   ├── *.ply                                        ← 房屋点云 (带语义)
│   ├── *.semseg.json / *.fsegs.json                 ← 语义分割标签
│   └── panorama_to_region.txt                       ← 全景图→房间映射
├── matterport_camera_intrinsics.zip                  ← 原始扫描时的相机内参
├── matterport_camera_poses.zip                       ← 原始扫描相机位姿
├── cameras.zip
└── tmpb9j0xo4w                                       ← 残留构建临时文件 (可忽略)
```

### 9.3 config.yaml 内容 (示例)

```yaml
asset_path: /ssd/share/Matterport3D/data/v1/scans/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1.obj
usd_dir: /ssd/share/Matterport3D/data/v1/scans/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1
usd_file_name: isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1.usd
force_usd_conversion: true
make_instanceable: false
collision_props:
  collision_enabled: true
collision_approximation: convexDecomposition
# Generated by MP3DMeshConverter on 2024-08-13 at 16:37:38.
```

→ InternRobotics 还顺带生成过 `isaacsim_*.usd` 用于 Isaac Sim，但 obj 本身可以直接喂给 BlenderProc / Habitat-Sim / Open3D / Trimesh / Blender。

---

## 10. Mesh vs Texture 概念厘清

| 概念 | 是什么 | 在 MP3D 目录里对应 |
|---|---|---|
| **Mesh (网格)** | 3D 几何：顶点 (vertex) + 面 (face) + UV 坐标。**这就是用来做碰撞检测、光线相交、深度计算的几何体** | `bed1a77d92d64f5cbbaaae4feed64ec1.obj` (111K 顶点 / 216K 面) |
| **Texture (纹理)** | 2D 图片，贴在 mesh 表面让它"看起来像真的"。只影响 RGB 颜色，不影响几何/碰撞 | `textures/*.jpg` (23 张 jpg) |
| **Material (材质)** | 描述 mesh 的哪一块 face 用哪一张 texture，以及反射/漫反射等参数 | `*.mtl` |

**形象比喻**：
- **mesh** = 一座建筑的"骨架/墙体" — 决定了"哪里有东西、哪里是空的"
- **texture** = 贴在墙体上的"墙纸/瓷砖照片" — 只决定"墙看起来什么颜色花纹"
- **material** = "墙纸用法说明书" — 哪面墙贴哪张图

`.obj` 文件格式 (ASCII 文本)：
```
v  1.234  2.567  -0.891         <- 顶点坐标
vt 0.234 0.567                   <- UV 纹理坐标
vn 0.0 1.0 0.0                   <- 法线
f 1/1/1  2/2/2  3/3/3            <- 面 (用三个顶点编号定义一个三角形)
```

**做不同任务时使用的部分：**
- **碰撞检测** → 直接对 `.obj` 的三角面做 AABB / BVH 查询 (trimesh、Open3D、PyBullet、Habitat-Sim 都内置)
- **NavDP 论文里的 ESDF + A\*** → 把 `.obj` 体素化得到 0.05m voxel grid，再算 ESDF
- **fisheye 重渲染** → BlenderProc 加载 `.obj`，相机切 fisheye，纹理由 `.mtl` + `textures/*.jpg` 自动应用
- **语义条件 NavDP / VLN 任务** → 用 `house_segmentations/*.ply` (带语义标签的 PLY mesh)

---

## 11. 6 个场景数据集下载指南

目前已下载 MP3D (1/6)。剩余 5 个，按"申请难度从低到高"排序：

| #   | 数据集          | 难度                  | 获取方式                                                                                        | 需要拿的具体内容                                                                                                                         |
| --- | ------------ | ------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Replica**  | ⭐ 直接下载              | https://github.com/facebookresearch/Replica-Dataset                                         | `cd Replica-Dataset && ./download.sh` (约 100GB)，18 个 scene 的 mesh + texture                                                      |
| 2   | **HSSD**     | ⭐⭐ HF 接受 ToS        | https://huggingface.co/datasets/hssd/hssd-hab                                               | 网页点 "Agree" → `git lfs clone`；约 211 个高质量合成场景，glb 格式                                                                              |
| 3   | **HM3D**     | ⭐⭐ 学术 license       | https://aihabitat.org/datasets/hm3d/ → 邮件签 form → Matterport 邮件回链接                          | 1000 个真实扫描，下 `hm3d-train-glb-v0.2.tar` (~140GB)                                                                                  |
| 4   | **3D-FRONT** | ⭐⭐⭐ 阿里天池签 ToS       | https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset → 下 PDF 签字 → 邮件回三个链接 | **必须三件套都要**：① 3D-FRONT (场景 JSON) ② 3D-FUTURE-model (家具 mesh, 几十 GB) ③ 3D-FRONT-textures (材质贴图)。缺一不可。已有 chunk `00154c06-...` 就用这个 |
| 5   | **Gibson**   | ⭐⭐⭐ Stanford 签 EULA | https://github.com/StanfordVL/GibsonEnv → 填 form → 邮件回链接                                    | 拿 Habitat 兼容版 `gibson_habitat_trainval` (约 27GB)，572 个 scene 的 glb                                                               |

### 11.1 建议下载顺序

- **优先级**：如果目标是"先跑通一个 fisheye 重渲染 demo"：
  1. **Replica 最快** (10 分钟下完，质量极高，18 个场景够验证 pipeline)
  2. 然后选已有 chunk 对应的 **3D-FRONT** 把已有轨迹真正打通端到端

- **HM3D 和 Gibson 申请慢** (邮件回复要 1-3 天)，可以先发申请邮件，期间用 Replica + HSSD 跑通流程

- **存储预算**：6 个数据集合计 ≈ **400 GB+**
  - HM3D 最大约 140GB
  - 3D-FRONT/HSSD 合计约 100GB
  - Gibson ~27GB
  - Replica ~100GB
  - MP3D 已下 41GB

### 11.2 格式不统一

| 数据集 | mesh 格式 |
|---|---|
| MP3D | `.obj` |
| 3D-FRONT | `.json + .obj` |
| HSSD | `.glb` |
| HM3D | `.glb` |
| Replica | `.ply` |
| Gibson | `.glb` |

BlenderProc 都支持，但建议写个**统一的 mesh loader 抽象层**便于后续 fisheye pipeline 复用。

### 11.3 最小验证路径 (推荐)

```
Day 1:  申请 HM3D + Gibson + 3D-FRONT 邮件
Day 1:  下 Replica (一键脚本) → 写一个 BlenderProc fisheye 加载脚本，验证流程
Day 1:  下 HSSD (HF 即时给) → 验证脚本兼容 .glb
Day 2-3: 收到 3D-FRONT 链接 → 下 3 件套 → 用已有 vln_n1 chunk + fisheye 跑通端到端
Day 3-7: 拿到 HM3D / Gibson → 全量补齐
```

完成后就能拿同样的 200K 轨迹**重新生成属于你自己的 fisheye 训练集**，与论文的 D435i / Zed 版本并列成第三种。

---

## 12. fisheye 重渲染技术路线

### 12.1 推荐做法 (按目标排序)

1. **首选 BlenderProc** (NavDP 用的就是这个，鱼眼最简单)：
   - `bproc.camera.set_intrinsics_from_blender_params(lens, ..., lens_unit='FOV')` 配 `panorama_type='EQUIRECTANGULAR'` 或 Cycles 的 `PANO_FISHEYE_EQUIDISTANT / EQUISOLID`
   - `bproc.camera.add_camera_pose(T_cam_world[i])` 逐帧
   - 加载场景 .blend / .obj
   - 这条路能完美复刻论文的渲染域，仅替换内参为 fisheye

2. **Habitat-Sim**：原生支持 `equirectangular / fisheye sensor`，但仅用于 mp3d / hm3d / replica / gibson 这类 Habitat 兼容场景

3. **Isaac Sim** (如果场景资产是 .usd，比如 InternScenes)：内置 `fisheye_polynomial` 投影模型，`Camera.set_projection_type("fisheyePolynomial")`

### 12.2 几个坑提醒

- **z=0.357m** 是相机距地面高度，已 baked 在 extrinsic 里。如果模拟更高/更矮的鱼眼相机机器人，需要改这一项
- `extrinsic` 矩阵的旋转部分是 `Rx(-90°)` (OpenGL 风格 Y-up → ROS 风格 Z-up 的转换)，重渲染前用 1 帧的 RGB ground truth 做"原内参重渲染对齐"以确认坐标系约定
- 帧率 30 fps，但相邻帧机器人只移动 ~3.7 cm，可以**子采样**节省渲染时间
- `meta/pointcloud.ply` 可以当作占用栅格 / 空间布局参考，但不要指望用它直接渲染出 NavDP 训练分布的 RGB

---

## 13. 关键文件路径速查表

### 13.1 NavDP 仓库 (`/ssd4/github-knowledge-base/NavDP/`)

```
NavDP/
├── README.md                                          ← 主文档
├── baselines/navdp/
│   ├── navdp_server.py                                ← 推理服务入口
│   ├── policy_agent.py
│   ├── policy_backbone.py                             ← RGBD/PixelGoal/ImageGoal backbones
│   ├── policy_network.py:9                            ← NavDP_Policy 主类
│   ├── depth_anything/                                ← DepthAnything 编码器
│   └── requirements.txt
├── baselines/{ddppo,gnm,iplanner,logoplanner,nomad,vint,viplanner}/  ← 其他 baseline
├── configs/{robots,scenes,tasks}/                     ← IsaacSim 评测配置
├── utils_tasks/                                       ← 评测辅助
├── wheeled_robots/controllers/                        ← 轮式机器人控制器
├── eval_{nogoal,pointgoal,imagegoal,startgoal}_wheeled.py  ← 评测入口
└── teleop_{nogoal,pointgoal,imagegoal}_wheeled.py     ← 遥操作入口
```

### 13.2 InternNav 仓库 (`/ssd4/github-knowledge-base/InternNav/`)

```
InternNav/
├── internnav/
│   ├── configs/
│   │   ├── model/navdp.py                             ← NavDP 模型配置
│   │   └── trainer/{exp,eval,il}.py                   ← 训练器配置
│   ├── dataset/
│   │   ├── navdp_lerobot_dataset.py:34                ← NavDP LeRobot 数据集 (训练用)
│   │   ├── navdp_dataset.py                           ← NavDP 旧版数据集
│   │   ├── cma_lerobot_dataset.py / rdp_lerobot_dataset.py  ← VLN-CE/PE 用
│   │   └── internvla_n1_lerobot_dataset.py
│   ├── model/
│   │   ├── basemodel/navdp/navdp_policy.py:34         ← NavDPNet (HF PreTrainedModel 包装)
│   │   ├── basemodel/internvla_n1/navdp.py:16         ← NavDP_Policy_DPT_CriticSum_DAT (双系统)
│   │   └── encoder/navdp_backbone.py                  ← RGBD/ImageGoal/PixelGoal backbones
│   ├── trainer/
│   │   ├── navdp_trainer.py:11                        ← NavDPTrainer.compute_loss
│   │   ├── base.py                                    ← BaseTrainer
│   │   └── internvla_n1_trainer.py
│   ├── agent/                                         ← 智能体 (含 InternVLA-N1)
│   ├── env/ evaluator/ habitat_extensions/ utils/
├── scripts/
│   ├── train/base_train/
│   │   ├── train.py:170,219                           ← 训练主入口 (调用 NavDPTrainer)
│   │   ├── start_train.sh                             ← 启动脚本 (--model navdp → 8 GPU torchrun)
│   │   └── configs/navdp.py                           ← NavDP 超参
│   ├── dataset_converters/
│   │   └── vlnce2lerobot.py                           ← VLN-CE → LeRobot 转换器
│   ├── iros_challenge/                                ← IROS 2025 Challenge
│   ├── notebooks/inference_only_demo.ipynb            ← InternVLA-N1 推理 demo
│   └── realworld/                                     ← 真实世界部署
├── docs/{changelog.md, compatibility.md}
└── third_party/
```

### 13.3 下载的数据

```
/ssd4/github-knowledge-base/00154c06-2ee2-408a-9664-b8fd74742897/   ← vln_n1 示例 chunk
└── (见 §6.1)

/ssd5/datasets/Scene-N1/mp3d_n1/                                    ← MP3D 全量场景资产
└── 90 个场景目录 (见 §9.2)
```

---

## 14. 关键结论速查

1. **NavDP 训练代码只在 InternNav 仓库**，启动命令 `bash scripts/train/base_train/start_train.sh --model navdp` (8 GPU torchrun)
2. **NavDP 训练数据只用 `vln_n1` 一个子集**，不用 `vln_ce` 也不用 `vln_pe`；后者给 InternNav 其他 VLN baselines 用
3. **InternData-N1 数据生成 pipeline 未开源** — 论文 III 节描述了方法 (ESDF + A\* + spline + BlenderProc)，但代码没放
4. **d435i / zed 是两套相机 FOV 配置** (69°×42° vs 110°×70°)，源出论文 III 节
5. **chunk `00154c06-...` 来自 3D-FRONT** (UUID 命中 Kaggle 上的 3D-FRONT scene 列表)，属于 `vln_n1/traj_data/3dfront_d435i/`
6. **一个 episode = 一条轨迹**，parquet 里 `action` 列 (4×4 SE(3)) 是逐帧机器人世界位姿
7. **轨迹本身相机无关**，可换任意 fisheye 内参重渲染，唯一缺的是带纹理场景 mesh
8. **Scene-N1 只覆盖 6 个源数据集中的 MP3D 一个** (license 限制)，其余 5 个需自己申请
9. **Scene-N1 的 `mp3d_n1/` 是 MP3D 全量 90 个场景** (mesh + texture + 语义标注)，NavDP 训练实际用了 68 个
10. **Mesh ≠ Texture**：`.obj` 是几何骨架 (用于碰撞/A\*/ESDF/渲染)，`textures/*.jpg` 是贴皮 (只影响颜色)，`.mtl` 是粘贴说明
