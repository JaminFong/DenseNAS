import torch
import torch.nn as nn

OPS = {
    'mbconv_k3_t1': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=1, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=6, affine=affine, track_running_stats=track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Skip(C_in, C_out, 1, affine=affine, track_running_stats=track_running_stats),
}


class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(ConvBNReLU, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class SepConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU6(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True, track_running_stats=True):
        super(MBConv, self).__init__()
        self.t = t
        if self.t > 1:
            self.mbconv = nn.Sequential(
                nn.Conv2d(C_in, C_in*self.t, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True),

                nn.Conv2d(C_in*self.t, C_in*self.t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*self.t, bias=False),
                nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True),

                nn.Conv2d(C_in*self.t, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            )
        else:
            self.mbconv = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True),

                nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(C_out),
            )

    def forward(self, x):
        out = self.mbconv(x)
        if out.shape == x.shape:
            return out + x
        else:
            return out


class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(Skip, self).__init__()
        if C_in!=C_out:
            skip_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
            stride = 1
        self.op=Identity(stride)

        if C_in!=C_out:
            self.op=nn.Sequential(skip_conv, self.op)

    def forward(self,x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride]
