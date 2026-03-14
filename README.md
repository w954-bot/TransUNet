# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## 📰 News
- [7/26/2024] TransUNet, which supports both 2D and 3D data and incorporates a Transformer encoder and decoder, has been featured in the journal Medical Image Analysis ([link](https://www.sciencedirect.com/science/article/pii/S1361841524002056)).
```bibtex
@article{chen2024transunet,
  title={TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and others},
  journal={Medical Image Analysis},
  pages={103280},
  year={2024},
  publisher={Elsevier}
}
```

- [10/15/2023] 🔥 3D version of TransUNet is out! Our 3D TransUNet surpasses nn-UNet with 88.11% Dice score on the BTCV dataset and outperforms the top-1 solution in the BraTs 2021 challenge and secure the second place in BraTs 2023 challenge. Please take a look at the [code](https://github.com/Beckschen/3D-TransUNet/tree/main) and [paper](https://arxiv.org/abs/2310.07781).


## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data (All data are available!)

All data are available so no need to send emails for data. Please use the [BTCV preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC data](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link).

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

### 5. Train with your own PNG dataset (e.g., esophageal endoscopy)

This repo's training dataloader expects `.npz` files with two keys: `image` and `label`, and a `train.txt` list file. `image` can be grayscale `(H, W)` or RGB `(H, W, 3)`.

1) Convert your `png` image/mask pairs to the expected format:

```bash
python tools/prepare_endoscopy_png_dataset.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_root /path/to/endoscopy_transunet \
  --train_ratio 0.8 \
  --binarize_mask
```

2) Launch training with the custom dataset mode:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset Custom \
  --root_path /path/to/endoscopy_transunet/train_npz \
  --list_dir /path/to/endoscopy_transunet/lists \
  --num_classes 2 \
  --vit_name R50-ViT-B_16
```

Notes:
- Use `--num_classes 2` for binary lesion/background segmentation.
- If your mask already stores class ids (0..N-1), remove `--binarize_mask` and set `--num_classes N`.
- RGB PNGs are kept as 3-channel arrays by default (no grayscale conversion).
- `val.txt` is required for per-epoch validation, and `best_model.pth` is saved by best validation Dice.
- For empty-foreground validation samples: gt/pred both empty is scored as Dice=1.0; gt empty but pred non-empty is Dice=0.0.

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations


```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
