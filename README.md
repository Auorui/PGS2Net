# PGS²Net

Code implementation of the **A Physics-Guided Stage-Aware Frequency Spectral Network for Remote Sensing Image Dehazing**

## Prerequisites
```bash
pip install -r requirements.txt
```
Install torch and torchvision first, then install pyzjr.

## Dataset

You can download the RSHD dataset on [Baidu Netdisk](https://pan.baidu.com/s/1mfpsgL1lf3EOOmN0ymlVfA?pwd=j3yb) 

You can download the downstream evaluation dataset on [Baidu Netdisk](https://pan.baidu.com/s/1BN7uLtRpbOaA103EWpHBPA?pwd=mjj3)

The partially synthetic cloud images are placed under `data/cloud_512/`. To reproduce the cloud synthesis described in the paper, run the following command **from the repository root**:

```bash
python ./script/cloud_fusion.py
```

## Model Training
To train a model from scratch, use
```python
python train.py --model "PGS2Net_s" --dataset_path "./data/RSHD/thick" --epochs 500 --batch_size 8
```
The names of supported models are detailed in MODEL_CLASSES within ./models/networks.py

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@ARTICLE{11806865,
  author={Zeng, Junrui and Chen, Bo and Zhang, Hua and Zhou, Huaifang and Miao, Hongrui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Physics-Guided Stage-Aware Frequency Spectral Network for Remote Sensing Image Dehazing}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote sensing image dehazing;Frequency spectral modeling;Physics-guided learning;Stage-aware network},
  doi={10.1109/TGRS.2026.3755826}}
```

## Contact
For questions about the code or dataset, please contact:

**2165648225@qq.com** (Gmail: zjricetea@gmail.com, may reply slowly)
