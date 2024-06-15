from torch import nn

class ConvLayer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel: int, 
        stride: int, 
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel, 
            stride=stride, 
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.4)

    
    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out

class ConvTransposeLayer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel: int, 
        stride: int, 
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel, 
            stride=stride, 
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.4)

    
    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()       
        self.conv1 = ConvLayer(in_channels, hidden_dim, kernel=3, stride=2)
        self.conv2 = ConvLayer(hidden_dim, hidden_dim//2, kernel=3, stride=1)
        self.conv_transpose1 = ConvTransposeLayer(hidden_dim//2, hidden_dim, kernel=3, stride=1)
        self.conv_transpose2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2)
        self.tanh = nn.Tanh()
        
    
    # forward method
    def forward(self, input): 
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv_transpose1(out)
        out = self.conv_transpose2(out)
        out = self.tanh(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, hidden_dim, kernel=3, stride=2)
        self.conv2 = ConvLayer(hidden_dim, hidden_dim//2, kernel=3, stride=2)
        self.conv3 = ConvLayer(hidden_dim//2, 1, kernel=3, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4, 1)
    
    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out