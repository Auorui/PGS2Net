import torch
from torch import nn


class ResnetBlock(nn.Module):

    def __init__(self, dim, down=True, first=False, levels=3, bn=False):
        super(ResnetBlock, self).__init__()
        blocks = []
        for i in range(levels):
            blocks.append(Block(dim=dim, bn=bn))
        self.res = nn.Sequential(
            *blocks
        ) if not first else None
        self.downsample_layer = nn.Sequential(
            nn.InstanceNorm2d(dim, eps=1e-6) if not bn else nn.BatchNorm2d(dim, eps=1e-6),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2)
        ) if down else None
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, 64, kernel_size=7),
            nn.InstanceNorm2d(64, eps=1e-6)
        ) if first else None

    def forward(self, x):
        if self.stem is not None:
            out = self.stem(x)
            return out
        out = x + self.res(x)
        if self.downsample_layer is not None:
            out = self.downsample_layer(out)
        return out


class Block(nn.Module):

    def __init__(self, dim, bn=False):
        super(Block, self).__init__()

        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim) if not bn else nn.BatchNorm2d(dim, eps=1e-6),
                       nn.LeakyReLU()]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim) if not bn else nn.BatchNorm2d(dim, eps=1e-6)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PALayer(nn.Module):

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.LeakyReLU(inplace=False) if relu else None
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if not bn else nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        if self.relu is not None:
            x = self.relu(x)
        x = self.pad(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Fusion_Block(nn.Module):

    def __init__(self, channel, bn=False, res=False):
        super(Fusion_Block, self).__init__()
        self.bn = nn.InstanceNorm2d(channel, eps=1e-5, momentum=0.01, affine=True) if not bn else nn.BatchNorm2d(
            channel, eps=1e-5, momentum=0.01, affine=True)
        self.merge = nn.Sequential(
            ConvBlock(channel, channel, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        ) if not res else None
        self.block = ResnetBlock(channel, down=False, levels=2, bn=bn) if res else None

    def forward(self, o, s):
        o_bn = self.bn(o) if self.bn is not None else o
        x = o_bn + s
        if self.merge is not None:
            x = self.merge(x)
        if self.block is not None:
            x = self.block(x)
        return x


class FE_Block(nn.Module):

    def __init__(self, plane1, plane2, res=True):
        super(FE_Block, self).__init__()
        self.dsc = ConvBlock(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1, relu=False)

        self.merge = nn.Sequential(
            ConvBlock(plane2, plane2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        ) if not res else None
        self.block = ResnetBlock(plane2, down=False, levels=2) if res else None

    def forward(self, p, s):
        x = s + self.dsc(p)
        if self.merge is not None:
            x = self.merge(x)
        if self.block is not None:
            x = self.block(x)
        return x


class Iter_Downsample(nn.Module):
    def __init__(self, ):
        super(Iter_Downsample, self).__init__()
        self.ds1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ds2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        return x, x1, x2


class CALayer(nn.Module):

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ConvGroups(nn.Module):

    def __init__(self, in_planes, bn=False):
        super(ConvGroups, self).__init__()
        self.iter_ds = Iter_Downsample()
        self.lcb1 = nn.Sequential(
            ConvBlock(in_planes, 16, kernel_size=(3, 3), padding=1), ConvBlock(16, 16, kernel_size=1, stride=1),
            ConvBlock(16, 16, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(16, 64, kernel_size=1, bn=bn, relu=False))
        self.lcb2 = nn.Sequential(
            ConvBlock(in_planes, 32, kernel_size=(3, 3), padding=1), ConvBlock(32, 32, kernel_size=1),
            ConvBlock(32, 32, kernel_size=(3, 3), padding=1), ConvBlock(32, 32, kernel_size=1, stride=1, bn=bn),
            ConvBlock(32, 32, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(32, 128, kernel_size=1, bn=bn, relu=False))
        self.lcb3 = nn.Sequential(
            ConvBlock(in_planes, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1, bn=bn),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(64, 256, kernel_size=1, bn=bn, relu=False))

    def forward(self, x):
        img1, img2, img3 = self.iter_ds(x)
        s1 = self.lcb1(img1)
        s2 = self.lcb2(img2)
        s3 = self.lcb3(img3)
        return s1, s2, s3

class PSMBNet(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(PSMBNet, self).__init__()
        # 下采样
        self.down1 = ResnetBlock(3, first=True)

        self.down2 = ResnetBlock(ngf, levels=2)

        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True)
        )

        # 上采样

        self.up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2),
        )

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
        )

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
        )

        self.info_up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf)  # if not bn else nn.BatchNorm2d(ngf, eps=1e-5),
        )

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())

        self.pa2 = PALayer(ngf * 4)
        self.pa3 = PALayer(ngf * 2)
        self.pa4 = PALayer(ngf)

        self.ca2 = CALayer(ngf * 4)
        self.ca3 = CALayer(ngf * 2)
        self.ca4 = CALayer(ngf)

        self.down_dcp = ConvGroups(3, bn=bn)

        self.fam1 = FE_Block(ngf, ngf)
        self.fam2 = FE_Block(ngf, ngf * 2)
        self.fam3 = FE_Block(ngf * 2, ngf * 4)

        self.att1 = Fusion_Block(ngf)
        self.att2 = Fusion_Block(ngf * 2)
        self.att3 = Fusion_Block(ngf * 4, bn=bn)

        self.merge2 = nn.Sequential(
            ConvBlock(ngf * 2, ngf * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.merge3 = nn.Sequential(
            ConvBlock(ngf, ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, hazy, img=0, first=True):
        if not first:
            dcp_down1, dcp_down2, dcp_down3 = self.down_dcp(img)
        x_down1 = self.down1(hazy)  # [bs, ngf, ngf * 4, ngf * 4]

        att1 = self.att1(dcp_down1, x_down1) if not first else x_down1

        x_down2 = self.down2(x_down1)  # [bs, ngf*2, ngf*2, ngf*2]
        att2 = self.att2(dcp_down2, x_down2) if not first else None
        fuse2 = self.fam2(att1, att2) if not first else self.fam2(att1, x_down2)

        x_down3 = self.down3(x_down2)  # [bs, ngf * 4, ngf, ngf]
        att3 = self.att3(dcp_down3, x_down3) if not first else None
        fuse3 = self.fam3(fuse2, att3) if not first else self.fam3(fuse2, x_down3)

        x6 = self.pa2(self.ca2(self.res(x_down3)))

        fuse_up2 = self.info_up1(fuse3)
        fuse_up2 = self.merge2(fuse_up2 + x_down2)

        fuse_up3 = self.info_up2(fuse_up2)
        fuse_up3 = self.merge3(fuse_up3 + x_down1)

        x_up2 = self.up1(x6 + fuse3)
        x_up2 = self.ca3(x_up2)
        x_up2 = self.pa3(x_up2)

        x_up3 = self.up2(x_up2 + fuse_up2)
        x_up3 = self.ca4(x_up3)
        x_up3 = self.pa4(x_up3)

        x_up4 = self.up3(x_up3 + fuse_up3)

        return x_up4


if __name__ == "__main__":
    from pyzjr.nn import summary_2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PSMBNet()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_2(model, input_size=(3, 256, 256))
