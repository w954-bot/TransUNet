# TransUNet 仓库代码架构梳理

## 1. 仓库总体分层

该仓库是一个以 **医学图像分割** 为目标的训练/推理工程，围绕 Synapse 数据集提供了完整流程，核心可以分为五层：

1. **入口层（CLI）**：`train.py`、`test.py`。
2. **训练编排层**：`trainer.py`。
3. **模型层**：`networks/` 下的 TransUNet 主体与配置。
4. **数据层**：`datasets/dataset_synapse.py` 与 `lists/` 索引文件。
5. **评估与工具层**：`utils.py`（DiceLoss、3D 体数据切片推理、指标计算与保存）。

---

## 2. 运行主链路（从命令到结果）

### 2.1 训练链路

- 从 `train.py` 解析参数，按 `dataset_config` 注入 Synapse 的路径与类别数。
- 构建实验名和 `snapshot_path`（编码了模型名、skip 数、patch 大小、epoch、batch size、学习率、seed 等）。
- 从 `CONFIGS_ViT_seg` 取模型配置，实例化 `VisionTransformer`，并加载 ViT 预训练权重（`.npz`）。
- 分发到 `trainer_synapse(args, net, snapshot_path)` 执行训练。

### 2.2 训练循环细节（`trainer.py`）

- 数据：`Synapse_dataset(split="train") + RandomGenerator`。
- 损失：`CrossEntropyLoss + DiceLoss`，等权重相加。
- 优化器：`SGD(momentum=0.9, weight_decay=1e-4)`。
- 学习率：多项式衰减 `lr = base_lr * (1 - iter/max_iter)^0.9`。
- 日志：TensorBoard + 文本日志。
- 存储：后半程按间隔存 checkpoint，最后 epoch 强制保存。

### 2.3 推理链路（`test.py`）

- 与训练类似，通过 `dataset_config` 注入测试集路径和类别数。
- 复用与训练一致的 `snapshot_path` 规则定位模型参数。
- 调 `inference()` 遍历体数据，内部使用 `test_single_volume()` 做逐切片推理。
- 计算每类 Dice / HD95，汇总日志；可选保存 nii.gz 预测结果。

---

## 3. 模型层架构（TransUNet）

核心模型在 `networks/vit_seg_modeling.py`，可理解为：

`输入 -> (可选 ResNetV2 混合嵌入) -> ViT Encoder -> U-Net 风格 Decoder -> Segmentation Head`

### 3.1 编码器（Transformer）

- `Embeddings`：
  - 纯 ViT 模式：按 patch 切分卷积投影。
  - Hybrid 模式：先经过 `ResNetV2` 提取多尺度特征，再做 patch embedding。
- `Encoder`：由多层 `Block(Attention + MLP + 残差 + LayerNorm)` 组成。
- `Transformer.forward` 返回：
  - token 编码结果 `encoded`
  - 注意力权重（可视化时）
  - hybrid 模式下的多尺度 `features`（供 decoder skip-connection）

### 3.2 解码器（U-Net 风格）

- `DecoderCup` 先将 token 还原成二维特征图，再经 4 个 `DecoderBlock` 逐级上采样。
- `n_skip` 控制使用多少级 skip 特征，低于阈值的 skip 通道会被置 0。
- `SegmentationHead` 用卷积输出类别通道图。

### 3.3 混合骨干（ResNetV2）

`networks/vit_seg_modeling_resnet_skip.py` 提供预激活 ResNetV2：

- `StdConv2d`：卷积核标准化。
- `PreActBottleneck`：GN + Conv 的 pre-activation block。
- `ResNetV2.forward` 额外返回多尺度 feature 列表（倒序），用于 decoder 的 skip。

### 3.4 配置系统

`networks/vit_seg_configs.py` 通过 `ml_collections.ConfigDict` 统一描述：

- patch 大小、hidden size、层数、head 数。
- decoder 通道、skip 通道、类别数。
- 预训练权重路径（如 `R50+ViT-B_16.npz`）。

---

## 4. 数据与增强层

`datasets/dataset_synapse.py` 的关键点：

- `Synapse_dataset`：
  - 训练 split：读取 `.npz` 的 2D 切片 `image/label`。
  - 测试 split：读取 `.npy.h5` 的 3D 体数据。
- `RandomGenerator`：
  - 随机旋转/翻转增强。
  - 尺寸缩放到网络输入大小。
  - 输出 tensor：`image` 形状 `[1, H, W]`，`label` 为整型 mask。

`lists/lists_Synapse/*.txt` 提供训练/测试样本 ID 清单，是数据集读取的索引入口。

---

## 5. 损失、评估与结果保存

`utils.py` 负责训练/推理阶段复用能力：

- `DiceLoss`：多类别 one-hot Dice。
- `calculate_metric_percase`：二值 Dice 与 HD95（基于 medpy）。
- `test_single_volume`：
  - 对 3D 体按 z 维逐切片推理；必要时 resize。
  - 将预测恢复原尺度。
  - 可将预测/图像/标注写为 nii.gz（SimpleITK）。

---

## 6. 目录与职责速查

- `train.py`：训练入口、配置注入、模型初始化、启动 trainer。
- `trainer.py`：训练循环、loss/优化器、日志、checkpoint。
- `test.py`：测试入口、模型加载、全量评估。
- `networks/vit_seg_modeling.py`：TransUNet 主干 + 解码器 + 权重加载。
- `networks/vit_seg_modeling_resnet_skip.py`：Hybrid 模式 ResNetV2。
- `networks/vit_seg_configs.py`：模型配置模板。
- `datasets/dataset_synapse.py`：数据集与增强。
- `utils.py`：DiceLoss、推理后处理、指标、NIfTI 保存。
- `lists/lists_Synapse/`：样本索引文件。

---

## 7. 可扩展点（落地改造建议）

1. **新增数据集**：仿照 `dataset_config`（`train.py`/`test.py`）+ 新 Dataset 类 + 新 list 文件。
2. **新增模型变体**：在 `vit_seg_configs.py` 注册配置，并在 `CONFIGS` 字典暴露。
3. **训练策略升级**：在 `trainer.py` 中替换优化器、调度器、损失权重策略。
4. **评估指标扩展**：在 `utils.py` 新增 per-class 指标并在 `test.py` 汇总输出。

