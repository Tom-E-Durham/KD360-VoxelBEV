# KD360-VoxelBEV

Official codes for KD360-VoxelBEV.

## 1) Clone Repository

```bash
git clone git@github.com:Tom-E-Durham/KD360-VoxelBEV.git
cd KD360-VoxelBEV
```

## 2) Install Dependencies

Create and activate your own Python environment (Conda or venv), then run:

```bash
pip install -r requirements.repro.txt
pip install -e .
```

`pip install -e .` installs the local `fisheye_tools` package, which is used in the dataset pipeline to convert dual-fisheye images to equirectangular images.

## 3) Version Reference

Reference environment used for training:

- CUDA: `11.8`
- PyTorch: `2.4.0+cu118`
- torchvision: `0.19.0`
- torchaudio: `2.4.0`
- numpy: `1.24.4`
- opencv-python: `4.10.0.84`
- efficientnet-pytorch: `0.7.1`
- einops: `0.8.1`
- tensorboardX: `2.6.2.2`

## 4) Dataset and Checkpoint

Dataset and teacher checkpoint links:

- Extended Dur360BEV dataset (Hugging Face): [https://huggingface.co/datasets/TomEeee/Dur360BEV-Extended](https://huggingface.co/datasets/TomEeee/Dur360BEV-Extended)

- Teacher checkpoint (ResNet101 backbone, Google Drive): [https://drive.google.com/file/d/1_cbgx4DoA_vmeKOoAcDQAqZdzc2WJrld/view?usp=sharing](https://drive.google.com/file/d/1_cbgx4DoA_vmeKOoAcDQAqZdzc2WJrld/view?usp=sharing)

After download:

```bash
export DUR360BEV_DATASET_DIR=/path/to/Dur360BEV_Dataset_Complete
export LOAD_CKPT=/path/to/teacher.pth
```

## 5) Run Training (Distilled Student)

From repository root:

```bash
LOAD_CKPT=/path/to/teacher.pth bash bash/train.sh
```

Default settings in `bash/train.sh`:

- `backbone='effb0_ori'`: student backbone in distillation training.
- `stage`: distillation feature location. 
  - `stage1`: after feature fusion (or, without gated fusion, the corresponding location in the LiDAR branch).
  - `stage2`: feature map at the output of the U-Net encoder.
  - `stage3`: feature map at the output of the U-Net decoder.
- `dis_type='cwd'`: distillation loss type.
- `dataset_name='Dur360BEV'`: dataset source.
- `dataset_version`: use `extended` (default: `extended`).

## 6) Evaluate Distilled Student

Unified evaluation script:

```bash
CKPT_PATH=/path/to/student.pth bash bash/eval_student.sh
```

Required argument:

```bash
CKPT_PATH=/path/to/student.pth bash bash/eval_student.sh
```

Common options:

```bash
# IoU evaluation (default)
CKPT_PATH=/path/to/student.pth MODE=eval DATASET_VERSION=extended bash bash/eval_student.sh

# Speed benchmark
CKPT_PATH=/path/to/student.pth MODE=speed REPEATS=30 bash bash/eval_student.sh
```

Notes:

- `MODE`: `eval` or `speed`
- `DATASET_VERSION`: use `extended`
- `DUR360BEV_DATASET_DIR`: set this env var if dataset is not in the default path
