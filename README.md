# PGS²Net
PGS²Net

Due to the length of the paper, we have abandoned some experimental demonstrations, but we have provided the complete experimental content in our Github repository


## Effect of Model Width and Depth
We use the –s model as the baseline to study the effect of model width and depth.
Among all configurations, Exp5 provides the best performance–efficiency trade-off and is adopted as the –b model in the following experiments.
| Model Number   |     embed_dim     | depth       |  PSNR/SSIM  |  Parameters(M)  | GFLOPs  | Note          |
|:----------:|:-----------------:|:-------------:|:-------------:|:------------:|:----------------:|--------------|
| -s       | [24,48,96,48,24]  | [2,2,4,2,2] | 26.06/0.870 |    2.34    |     43.14      | Baseline-s model       |
| Exp1     | [32,64,128,64,32] | [2,2,4,2,2] | 26.16/0.873 |    4.04    |     72.82      | Increase the width       |
| Exp2     | [48,96,192,96,48] | [2,2,4,2,2] | 26.81/0.878 |    8.84    |     155.13     | Further increase the width   |
| Exp3     | [24,48,96,48,24]  | [1,1,2,1,1] | 25.02/0.861 |    1.20    |     22.13      | Reduce depth       |
| Exp4     | [24,48,96,48,24]  | [4,4,8,4,4] | 26.20/0.873 |    4.62    |     85.17      | Increase depth     |
| Exp5     | [32,64,128,64,32] | [4,4,8,4,4] | 26.69/0.877 |    7.99    |     143.76     | Width and depth increase simultaneously    |


## Ablation Study

### Loss Function
Our loss ablation study was initially built upon the L1 + Contrast loss.
By further incorporating the frequency loss and systematically ablating each component, we observe that the L1 + Frequency combination consistently outperforms other variants.
In contrast, adding the contrast loss leads to noticeable performance degradation, indicating that contrast enhancement objectives may be incompatible with accurate restoration in dehazing scenarios.
| L1 Loss | Frequency | Contrast | PSNR / SSIM |
|:-------:|:---------:|:--------:|:-----------:|
| ✓ | ✗ | ✗ | 25.68 / 0.860 |
| ✗ | ✓ | ✗ | 25.56 / 0.867 |
| ✓ | ✓ | ✗ | **26.06 / 0.870** |
| ✓ | ✗ | ✓ | 24.99 / 0.844 |
| ✓ | ✓ | ✓ | 25.49 / 0.863 |

