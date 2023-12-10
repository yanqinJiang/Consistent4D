import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanFilter(nn.Module):
    def __init__(self, filter_size):
        super(MeanFilter, self).__init__()
        self.filter_size = filter_size

    def forward(self, input_tensor):
        N, C, H, W = input_tensor.size()
        reshaped_tensor = input_tensor.reshape(N, C * H * W)
        filtered_tensor = self.mean_filter_1d(reshaped_tensor)
        filtered_tensor = filtered_tensor.reshape(N, C, H, W)
        return filtered_tensor

    def mean_filter_1d(self, signal):
        N, D = signal.size()
        filter_weights = torch.ones(1, 1, self.filter_size).to(signal.device) / self.filter_size
        padded_signal = F.pad(signal.transpose(1, 0).unsqueeze(1), (self.filter_size // 2, self.filter_size // 2), mode='replicate')
        filtered_signal = F.conv1d(padded_signal, filter_weights, groups=1)
        return filtered_signal.squeeze(1).transpose(1, 0)

class MeanFilterModule(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(MeanFilterModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        # 特征升维模块
        # self.up_module = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 特征降维模块
        self.out_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        )

        # 均值滤波模块
        self.mean_filter = MeanFilter(filter_size=filter_size)

    def forward(self, input_tensor, mode=1):
        # 特征升维
        if mode > 1:
            # 均值滤波
            filtered_features = self.mean_filter(input_tensor)
        else:
            filtered_features = input_tensor
            
        # 特征降维
        out_features = self.out_module(filtered_features)
        out_features += input_tensor

        return out_features

class InterpolateModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InterpolateModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x, interpolate=False):
        input = x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        if False:
            assert x.shape[0] == 3, "experimental code, only support batch_size=3"
            mask1 = torch.Tensor([1., 0., 1.]).reshape(3, 1, 1, 1).to(x)
            mask2 = torch.Tensor([0., 1.0, 0.]).reshape(3, 1, 1, 1).to(x)
            x = x * mask1
            x = x + x.mean(0, keepdim=True) * mask2
            x[1] += (x[0] + x[2]) / 2 # interpolate between 0 and 2

        # output = self.conv2(x)
        output = x
        # output = input + x
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.mapping_in = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
        )
        self.mapping_out = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # zero_init
        nn.init.zeros_(self.mapping_out[-1].weight)
        
    def forward(self, x, **kwargs):
        input = x
        x = self.mapping_in(x)
        x = self.mapping_out(x)

        output = input + x
        output = output.clamp(0, 1)
        return x  


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_head=4, n_patches=1024):
        super(TransformerBlock, self).__init__()

        self.n_patches = n_patches
        self.mapping_in = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )
        self.mapping_out = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, out_channels))

        self.norm2 = nn.LayerNorm(out_channels)
        self.attn2 = nn.MultiheadAttention(out_channels, n_head, bias=True)
    
    def forward(self, x, interpolate=False):
        B, C, H, W = x.shape
        x = self.mapping_in(x)
        x = x.reshape(B, -1, H*W).permute(0, 2, 1) # (N, L, C)
        
        if interpolate:
            x_extra_prev = torch.cat([x[0:1], x[:-1]], dim=0)
            x_extra_after = torch.cat([x[1:], x[-1:]], dim=0)
        else:
            x_extra_prev = x.clone()
            x_extra_after = x.clone()

        x_extra = torch.cat([x_extra_prev, x_extra_after], dim=1)

        x = x + self.pos_embedding
        y = x_extra + torch.cat([self.pos_embedding]*2, dim=1)

        x_residual = x
        x = self.norm2(x)
        y = self.norm2(y)

        x = x.permute(1, 0, 2)  # (N, L, C) -> (L, N, C)
        y = y.permute(1, 0, 2).detach()

        x_out, _ = self.attn2(x, y, y)
        x_out = x_out.permute(1, 0, 2)  # (L, N, C) -> (N, L, C)
        x_out = x_out + x_residual

        B, HW, C = x_out.shape
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_out = self.mapping_out(x_out)

        return x_out

class InterpolateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InterpolateBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.mapping_in = nn.Sequential(
                nn.Conv2d(in_channels*3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
        )
        self.mapping_out = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, interpolate=False):

        B, C, H, W = x.shape
        if interpolate:
            x_extra_prev = torch.cat([x[0:1], x[:-1]], dim=0)
            x_extra_after = torch.cat([x[1:], x[-1:]], dim=0)
        else:
            x_extra_prev = x.clone()
            x_extra_after = x.clone()

        input_tensor = torch.cat([x, x_extra_prev, x_extra_after], dim=1)
        out = self.mapping_in(input_tensor)
        out = self.mapping_out(out)

        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*(2**scale_factor), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, 3, kernel_size=3, stride=1, padding=1), # fixed
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        out = self.out(x)

        return x, out

class UpsampleCascadeModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, block_type="upsample", scale_factor=2):
        super(UpsampleCascadeModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        block_list = []
        for i in range(depth):
            block_list.append(UpsampleBlock(in_channels if i==0 else out_channels, out_channels, scale_factor))
        
        self.blocks = nn.ModuleList(block_list)
    
    def forward(self, x, **kwrags):
        return_list = []
        for i in range(self.depth):
            x, out = self.blocks[i](x)
            return_list.append(out)
        
        return return_list

class CascadeModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, block_type='residual'):
        super(CascadeModule, self).__init__()
        self.depth = depth
        block_list = []
        for i in range(depth):
            if block_type == 'residual':
                block_list.append(ResidualBlock(in_channels, out_channels))
            elif block_type == 'interpolate':
                block_list.append(InterpolateBlock(in_channels, out_channels))
            elif block_type == "transformer":
                block_list.append(TransformerBlock(in_channels, out_channels))
        
        self.blocks = nn.ModuleList(block_list)

    def forward(self, x, interpolate=False):
        batch_size = x.shape[0]
        return_list = []
        for i in range(self.depth):
            x = self.blocks[i](x[:batch_size], interpolate=interpolate)
            return_list.append(x)
        
        return return_list

if __name__ == '__main__':
    # 示例用法
    N = 4
    C = 4  # 输入特征通道数
    H = 16
    W = 16

    # 创建特征模块
    feature_module = FeatureModule(in_channels=C, out_channels=8, filter_size=3)

    # 创建输入特征
    input_tensor = torch.randn(N, C, H, W)

    # 进行特征升维降维操作
    output_tensor = feature_module(input_tensor, mode=2)

    print("输入特征大小:", input_tensor.size())
    print("输出特征大小:", output_tensor.size())