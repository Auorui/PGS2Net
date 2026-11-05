# PGS²Net

## Prerequisites
```bash
pip install -r requirements.txt
```
Install torch and torchvision first, then install pyzjr.

## Model Training
To train a model from scratch, use
```python
python train.py --model "PGS2Net_s" --dataset_path "./data/RSHD/thick" --epochs 500 --batch_size 8
```
The names of supported models are detailed in MODEL_CLASSES within ./models/networks.py

## Experiments

### Ablation Study
| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|
| **w/o RPF** | | | |
| Seed 11 | 24.9608 | 0.8870 | 0.1529 |
| Seed 12 | 24.9804 | 0.8883 | 0.1506 |
| Seed 13 | 25.0281 | 0.8885 | 0.1492 |
| Mean ± Std | 24.9898 ± 0.035 | 0.8879 ± 0.0008 | 0.1509 ± 0.0019 |
| **Encoder, first block** | | | |
| Seed 11 | 25.0378 | 0.8877 | 0.1503 |
| Seed 12 | 25.0111 | 0.8884 | 0.1494 |
| Seed 13 | 25.0742 | 0.8891 | 0.1507 |
| Mean ± Std | 25.0410 ± 0.032 | 0.8884 ± 0.0007 | 0.1501 ± 0.0007 |
| **Decoder, first block** | | | |
| Seed 11 | 25.0706 | 0.8877 | 0.1536 |
| Seed 12 | 25.1030 | 0.8895 | 0.1530 |
| Seed 13 | 25.0662 | 0.8879 | 0.1524 |
| Mean ± Std | 25.0799 ± 0.020 | 0.8884 ± 0.0010 | 0.1530 ± 0.0006 |
| **Random init** | | | |
| Seed 11 | 25.1409 | 0.8886 | 0.1507 |
| Seed 12 | 25.2097 | 0.8897 | 0.1504 |
| Seed 13 | 25.1860 | 0.8900 | 0.1496 |
| Mean ± Std | 25.1789 ± 0.035 | 0.8894 ± 0.0007 | 0.1502 ± 0.0006 |
| **Encoder, first block (Ours)** | | | |
| Seed 11 | 25.3065 | 0.8906 | 0.1503 |
| Seed 12 | 25.2535 | 0.8894 | 0.1495 |
| Seed 13 | 25.3273 | 0.8909 | 0.1477 |
| Mean ± Std | **25.2958 ± 0.038** | **0.8903 ± 0.0008** | **0.1492 ± 0.0013** |

| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|
| w/o RPF | 24.9898 ± 0.035 | 0.8879 ± 0.0008 | 0.1509 ± 0.0019 |
| Encoder, first block | 25.0410 ± 0.032 | 0.8884 ± 0.0007 | 0.1501 ± 0.0007 |
| Decoder, first block | 25.0799 ± 0.020 | 0.8884 ± 0.0010 | 0.1530 ± 0.0006 |
| Random init | 25.1789 ± 0.035 | 0.8894 ± 0.0007 | 0.1502 ± 0.0006 |
| **Encoder, first block (Ours)** | **25.2958 ± 0.038** | **0.8903 ± 0.0008** | **0.1492 ± 0.0013** |
