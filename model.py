import torch
from torch import nn
import torchvision
import math
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=32):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bnorm1(x)
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        x = x + time_emb
        x = self.bnorm2(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    
class UNet_Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x, t):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x, t)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    
class UNet_Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]) 
        
    def forward(self, x, t, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x, t)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(self, enc_chs, dec_chs, num_class, retain_dim, out_size, time_emb_dim):
        super().__init__()
        self.encoder = UNet_Encoder(enc_chs)
        self.decoder = UNet_Decoder(dec_chs)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.out_size = out_size
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x, t):
        t = self.time_mlp(t)
        enc_ftrs = self.encoder(x, t)
        out = self.decoder(x=enc_ftrs[::-1][0], encoder_features=enc_ftrs[::-1][1:], t=t)
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, tuple(self.out_size))
        return out