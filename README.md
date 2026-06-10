# PGS²Net
PGS²Net

Due to the length of the paper, we have abandoned some experimental demonstrations, but we have provided the complete experimental content in our Github repository

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
