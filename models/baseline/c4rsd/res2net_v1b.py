# res2net.py
# Res2Net backbone, adapted from https://github.com/Res2Net/Res2Net-PretrainedModels
import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

MODEL_URLS = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs, bns = [], []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Res2Net(nn.Module):
    """Full Res2Net with classification head (used only for loading pretrained weights)."""

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        super(Res2Net, self).__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class Res2Net_Encoder(nn.Module):
    """Res2Net encoder that returns multi-scale features (layer3, layer2, layer1, layer0)."""

    def __init__(self, block, layers, baseWidth=26, scale=4):
        super(Res2Net_Encoder, self).__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x_layer0 = x
        x = self.maxpool(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        return x_layer3, x_layer2, x_layer1, x_layer0


def build_res2net101_encoder(pretrained=True, verbose=True):
    """Build a Res2Net101 encoder with optional ImageNet-pretrained weights.

    Args:
        pretrained: whether to load ImageNet weights.
        verbose: print the number of matched keys.

    Returns:
        Res2Net_Encoder with loaded weights (if pretrained=True).
    """
    encoder = Res2Net_Encoder(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
    if not pretrained:
        if verbose:
            print("[Res2Net] pretrained=False, encoder initialized from scratch.")
        return encoder

    # Load pretrained full Res2Net101
    res2net101 = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
    state_dict = model_zoo.load_url(MODEL_URLS['res2net101_v1b_26w_4s'], map_location='cpu')
    res2net101.load_state_dict(state_dict)

    # Transfer matching keys
    pretrained_dict = res2net101.state_dict()
    model_dict = encoder.state_dict()
    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(matched)
    encoder.load_state_dict(model_dict)

    if verbose:
        total_model = len(model_dict)
        total_matched = len(matched)
        print(f"[Res2Net] Model keys: {total_model}, matched pretrained keys: {total_matched}")
        if total_matched == 0:
            print("[Res2Net][WARNING] No weights matched. Encoder is training from scratch.")
        elif total_matched < total_model:
            missing = [k for k in model_dict if k not in matched]
            print(f"[Res2Net][WARNING] {len(missing)} keys not covered. First few: {missing[:3]}")
        else:
            print("[Res2Net] All encoder weights loaded successfully.")

    return encoder

if __name__=="__main__":
    encoder = build_res2net101_encoder(pretrained=True, verbose=True)
    encoder.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256)
        x_layer3, x_layer2, x_layer1, x_layer0 = encoder(x)

    print(f"Input shape:    {tuple(x.shape)}")
    print(f"x_layer0 shape: {tuple(x_layer0.shape)}")  # [1, 64, 128, 128]
    print(f"x_layer1 shape: {tuple(x_layer1.shape)}")  # [1, 256, 64, 64]
    print(f"x_layer2 shape: {tuple(x_layer2.shape)}")  # [1, 512, 32, 32]
    print(f"x_layer3 shape: {tuple(x_layer3.shape)}")  # [1, 1024, 16, 16]