from models.baseline import *
from models.pgs2net import PGS2Net_s, PGS2Net_b


MODEL_CLASSES = {
    'PGS2Net_s':           PGS2Net_s,
    'PGS2Net_b':           PGS2Net_b,
    'FourierDehazeNet':    FourierDehazeNet,

    'DCP':                 DCP,                    # TPAMI 2010
    'CAP':                 CAP,                    # TIP 2015
    'DehazeNet':           DehazeNet,              # TIP 2016
    'AODNet':              AODNet,                 # ICCV 2017
    'GCANet':              GCANet,                 # WACV 2019
    'GridDehazeNet':       GridDehazeNet,          # ICCV 2019

    'FCTFNet':             FCTFNet,                # LGRS 2020
    'MSBDN_DFF':           MSBDN_DFF,              # CVPR 2020
    'MSBDN_RDFF':          MSBDN_RDFF,
    'FFANet':              FFANet,                 # AAAI 2020

    'MPRNet':              MPRNet,                 # CVPR 2021
    'AECRNet':             AECRNet,                # CVPR 2021
    'SwinIR':              SwinIR,                 # ICCVW 2021
    'FourKDehazing':       FourKDehazing,          # CVPR 2021

    'Uformer':             Uformer,                # CVPR 2022
    'DCINet':              DCINet,                 # TGRS 2022
    'Dehamer':             Dehamer,                # CVPR 2022
    'gunet_t':             gunet_t,                # arXiv 2022
    'gunet_s':             gunet_s,
    'gunet_b':             gunet_b,
    'gunet_d':             gunet_d,
    'TransRA':             TransRA,                # Multidimensional Systems and Signal Processing 2022
    'Restormer':           Restormer,              # CVPR 2022

    'PSMBNet':             PSMBNet,                # TGRS 2023
    'dehazeformer_t':      dehazeformer_t,         # TIP 2023
    'dehazeformer_s':      dehazeformer_s,
    'dehazeformer_b':      dehazeformer_b,
    'dehazeformer_d':      dehazeformer_d,
    'dehazeformer_w':      dehazeformer_w,
    'dehazeformer_m':      dehazeformer_m,
    'dehazeformer_l':      dehazeformer_l,
    'MB_TaylorFormer_B':   MB_TaylorFormer_B,      # ICCV 2023
    'MB_TaylorFormer_L':   MB_TaylorFormer_L,
    'MB_TaylorFormerV2_B': MB_TaylorFormerV2_B,    # TPAMI 2025
    'MB_TaylorFormerV2_L': MB_TaylorFormerV2_L,
    'SCANet':              SCANet,                 # CVPRW 2023
    'TrinityNet':          TrinityNet,             # TGRS 2023
    'AIDTramsformer':      AIDTramsformer,         # WACV 2023
    'C2PNet':              C2PNet,                 # CVPR 2023
    'ChaIR':               ChaIR,                  # KBS 2023

    'ASTA':                ASTA,                   # LGRS 2024
    'DEANet':              DEANet,                 # TIP 2024
    'MixDehazeNet_t':      MixDehazeNet_t,         # IJCNN 2024
    'MixDehazeNet_s':      MixDehazeNet_s,
    'MixDehazeNet_b':      MixDehazeNet_b,
    'MixDehazeNet_l':      MixDehazeNet_l,
    'SFHformer_t':         SFHformer_t,            # ECCV 2024
    'SFHformer_s':         SFHformer_s,
    'SFHformer_m':         SFHformer_m,
    'SFHformer_l':         SFHformer_l,
    'PhDnet_t':            PhDnet_t,               # IF 2024
    'PhDnet_s':            PhDnet_s,
    'PhDnet_b':            PhDnet_b,
    'PhDnet_d':            PhDnet_d,
    'RSDformer':           RSDformer,              # GRSL 2024
    'SFAN':                SFAN,                   # TGRS 2024
    'RSHazeNet':           RSHazeNet,              # ICASSP 2024
    'PCSformer':           PCSformer,              # TGRS 2024

    'decloudformer_t':     decloudformer_t,        # PR 2025
    'decloudformer_s':     decloudformer_s,
    'decloudformer_b':     decloudformer_b,
    'decloudformer_d':     decloudformer_d,
    'decloudformer_w':     decloudformer_w,
    'decloudformer_m':     decloudformer_m,
    'decloudformer_l':     decloudformer_l,
    'MABDT':               MABDT,                  # Signal Processing 2025
    'MCAFNet_l':           MCAFNet_l,              # TGRS 2025
    'DehazeXL':            DehazeXL,               # CVPR 2025
    'SFRDPNet':            SFRDPNet,               # TGRS 2025
    'C4RSD':               C4RSD,                  # TIP 2025
    'BMFHNet':             BMFHNet,                # TCSVT 2025

    'MemoryNet':           MemoryNet,              # TCE 2026
    'HPCDMNet':            HPCDMNet,               # TGRS 2026
}


def get_dehaze_networks(name):
    if name in MODEL_CLASSES:
        model = MODEL_CLASSES[name]()
        return model
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")

if __name__ == "__main__":
    import torch
    from pyzjr.nn import summary_2, model_complexity_info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_dehaze_networks('MCAFNet_l')
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_2(model, input_size=(3, 256, 256))
    model_complexity_info(model, input_size=(3, 256, 256))
