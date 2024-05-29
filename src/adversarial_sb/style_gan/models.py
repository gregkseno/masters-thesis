# import torch
# from torch import nn

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.identity_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, inputs):
#         x = inputs.clone().detach()
#         out = self.layer(x)
#         residual  = self.identity_map(inputs)
#         skip = out + residual
#         return self.relu(skip)
    

# class DownSampleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.MaxPool2d(2),
#             ResBlock(in_channels, out_channels)
#         )

#     def forward(self, inputs):
#         return self.layer(inputs)
    

# class UpSampleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
        
#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.res_block = ResBlock(in_channels + out_channels, out_channels)
        
#     def forward(self, inputs, skip):
#         x = self.upsample(inputs)
#         x = torch.cat([x, skip], dim=1)
#         x = self.res_block(x)
#         return x
    

# class Discriminator(nn.Module):
#     def __init__(self, in_channels: int = 3):
#         super().__init__()

#         def disc_block(in_filters, out_filters, normalization=True):
#             """Returns layers of each discriminator block"""
#             layers: list[nn.Module] = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalization:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.net = nn.Sequential(
#             *disc_block(in_channels, 64, normalization=False),
#             *disc_block(64, 128),
#             *disc_block(128, 256),
#             *disc_block(256, 512),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(512, 1)
#         )

#     def forward(self, x):
#         return self.net(x) 

# class Generator(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.encoding_layer1_ = ResBlock(in_channels, 64)
#         self.encoding_layer2_ = DownSampleConv(64, 128)
#         self.encoding_layer3_ = DownSampleConv(128, 256)
#         self.bridge = DownSampleConv(256, 512)
#         self.decoding_layer3_ = UpSampleConv(512, 256)
#         self.decoding_layer2_ = UpSampleConv(256, 128)
#         self.decoding_layer1_ = UpSampleConv(128, 64)
#         self.output = nn.Conv2d(64, out_channels, kernel_size=1)
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout2d(0.2)
        
#     def forward(self, x):
#         ###################### Enocoder #########################
#         e1 = self.encoding_layer1_(x)
#         e1 = self.dropout(e1)
#         e2 = self.encoding_layer2_(e1)
#         e2 = self.dropout(e2)
#         e3 = self.encoding_layer3_(e2)
#         e3 = self.dropout(e3)
        
#         ###################### Bridge #########################
#         bridge = self.bridge(e3)
#         bridge = self.dropout(bridge)
        
#         ###################### Decoder #########################
#         d3 = self.decoding_layer3_(bridge, e3)
#         d2 = self.decoding_layer2_(d3, e2)
#         d1 = self.decoding_layer1_(d2, e1)
        
#         ###################### Output #########################
#         output = self.tanh(self.output(d1))
#         return output


import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4,stride, 1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=features[0], kernel_size=4, stride=4, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, out_channels=feature, stride=3 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=4, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        return self.model(x).reshape(-1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_features = 64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features, out_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))