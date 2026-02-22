import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

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


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.calayer=CALayer(_in_channels)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.calayer(out)
        out = self.conv_1x1(out)
        out = out + x
        return out



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y = torch.sum(y, 1, keepdim=True) / 3
    y /= 4
    return y

class first_Net(nn.Module):
    def __init__(self):
        super(first_Net, self).__init__()
        # pre
        self.conv01 = nn.Conv2d(4, 16, 3, 1, 1)

        self.conv11 = RDB(16, 4, 16)
        self.conv12 = RDB(16, 4, 16)
        self.conv13 = RDB(16, 4, 16)

        self.conv20 = RDB(16, 4, 16)
        self.conv21 = BasicBlock(16, 16)
        self.conv22 = BasicBlock(16, 3)

    def forward(self, x):
        edge_x = edge_compute(x)
        x = torch.cat((x, edge_x), 1)

        x01 = self.conv01(x)

        x11 = self.conv11(x01)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x20 = x11 + x12 + x13
        x20 = self.conv20(x20)
        x21 = self.conv21(x20)
        x22 = self.conv22(x21)
        return (x22)


class second_Net(nn.Module):
    def __init__(self):
        super(second_Net, self).__init__()
        # pre
        self.conv01 = nn.Conv2d(6, 16, 3, 1, 1)

        self.conv11 = RDB(16, 4, 16)
        self.conv12 = RDB(16, 4, 16)
        self.conv21 = BasicBlock(16, 16)
        self.conv22 = BasicBlock(16, 3)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        x01 = self.conv01(x)

        x11 = self.conv11(x01)
        x12 = self.conv12(x11)

        x21 = self.conv21(x12)
        x22 = self.conv22(x21)
        return (x22)


class FCTFNet(nn.Module):
    def __init__(self):
        super(FCTFNet, self).__init__()
        self.conv01 = first_Net()
        self.conv02 = second_Net()

    def forward(self, x):
        first_out = self.conv01(x)
        second_out = self.conv02(x, first_out)
        # return (first_out, second_out)
        return second_out


if __name__ == "__main__":
    from pyzjr import summary_2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCTFNet()
    # model.load_state_dict(torch.load('best_metric_model.pth', map_location=torch.device('cuda:0')))
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_2(model, (3, 256, 256))