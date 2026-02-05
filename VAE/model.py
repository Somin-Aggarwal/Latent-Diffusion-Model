import torch
import torch.nn as nn
from torch.nn import init

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)
        self.initialize()
        
    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

'''
Xavier uniform initialization is used in deep NN
such that the weights are not too small or big and 
depend on the connections 
 U(-a,+a)
 a = gain * srqt(6/(fan_in+fan_out))
fan_in: The number of input connections to the layer (e.g., input channels × kernel size).
fan_out: The number of output connections.
The Goal: It attempts to keep the variance of the activations the same across every layer.
'''

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.initialize()
        
    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        b,c,h,w = x.shape
        x = nn.functional.interpolate(x,size=(2*h,2*w),mode="nearest")
        x = self.conv(x)
        return x

class SelfAttentionDiff(nn.Module):
    def __init__(self, in_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch

        self.group_norm = nn.GroupNorm(num_groups=min(in_ch,32), num_channels=in_ch)

        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        
        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))

        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
     
    def forward(self, x):
        
        # softmax(QK^T/sqrt(d_k))V
        b, c, h, w = x.shape
        
        x_dash = self.group_norm(x)

        q = self.proj_q(x_dash).view(b, c, -1) ##
        k = self.proj_k(x_dash).view(b, c, -1) # (b,c,h*w)
        v = self.proj_v(x_dash).view(b, c, -1) ##
        
        attn_weights = torch.bmm(q.permute(0,2,1), k) / (c ** 0.5)  # (b,h*w,h*w)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.bmm(v,attn_weights.permute(0,2,1))  # (b,h*w,edim)
        
        out = out.view(b, c, h, w) 
        out = self.proj_out(out)
               
        return x + self.gamma * out

'''
Pre Norm 
norm -> activation -> conv
'''

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn):
        super(ResBlock, self).__init__()
        
        self.gn1 = nn.GroupNorm(num_groups=min(in_channels,32), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.gn2 = nn.GroupNorm(num_groups=min(out_channels,32), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        if attn:
            self.attn = SelfAttentionDiff(in_ch=out_channels)
        else:
            self.attn = nn.Identity()
        
        self.silu = nn.SiLU()
            
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x):
                
        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)
        
        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        out = out + self.shortcut(x)
        out = self.attn(out)
        return out

class VAE(nn.Module):
    def __init__(self, img_ch, base_ch, ch_mul, attn, n_resblocks, e_ch):
        super().__init__()
        
        self.e_ch = e_ch
        self.img_ch = img_ch        
        self.initial_conv = nn.Conv2d(img_ch, base_ch, 3 ,1, 1)
        
        current_ch = base_ch
        self.encoder_blocks = nn.ModuleList()
        for i,mul in enumerate(ch_mul):
            out_ch = base_ch * mul
            for _ in range(n_resblocks):
                self.encoder_blocks.append(
                    ResBlock(current_ch,out_ch,attn=(i in attn))
                    )
                current_ch = out_ch
            if i != len(ch_mul) - 1:
                self.encoder_blocks.append(
                    Downsample(current_ch,out_ch)
                )
                current_ch = out_ch
        
        self.middleblocks = nn.ModuleList([
            ResBlock(current_ch, current_ch, attn=False),
            ResBlock(current_ch, current_ch, attn=False),
        ])
        
        self.quant_conv = nn.Conv2d(current_ch,2*e_ch,3,1,1)
        self.post_quant_conv = nn.Conv2d(e_ch,current_ch,3,1,1)
        
        self.decoder_blocks = nn.ModuleList()
        for i,mul in reversed(list(enumerate(ch_mul))):
            out_ch = base_ch * mul
            for _ in range(n_resblocks + 1):
                self.decoder_blocks.append(
                    ResBlock(current_ch,out_ch,attn=(i in attn))
                )
                current_ch = out_ch
            if i != 0:
                self.decoder_blocks.append(
                    Upsample(current_ch,out_ch)
                )
                current_ch = out_ch
        
        self.proj_out = nn.Sequential(
            nn.GroupNorm(num_groups=min(current_ch,32),num_channels=current_ch),
            nn.SiLU(),
            nn.Conv2d(current_ch, img_ch, 3, 1, 1)
        )
        self.initialize()
        
              
    def initialize(self):
        init.xavier_uniform_(self.initial_conv.weight)
        init.zeros_(self.initial_conv.bias)
        init.xavier_uniform_(self.proj_out[-1].weight, gain=1.5)
        init.zeros_(self.proj_out[-1].bias)
        init.xavier_uniform_(self.quant_conv.weight)
        init.zeros_(self.quant_conv.bias)
        init.xavier_uniform_(self.post_quant_conv.weight)
        init.zeros_(self.post_quant_conv.bias)
        
    def forward(self, x):
                
        x = self.initial_conv(x)
        
        for layer in self.encoder_blocks:
            x = layer(x)
        
        for layer in self.middleblocks:
            x = layer(x)
        
        moments = self.quant_conv(x)
        mean, logvar = torch.chunk(moments,2,dim=1)
        
        std = torch.exp(0.5*logvar)
        noise = torch.randn_like(std)
        
        latent_encoding = mean + std*noise
        
        x = self.post_quant_conv(latent_encoding)
        for layer in self.decoder_blocks:
            x = layer(x)
        
        x = self.proj_out(x)
        return x, mean, logvar, latent_encoding

def param_count(model):
    param_count = 0
    for params in model.parameters():
        param_count += params.numel()
    return param_count


if __name__=="__main__":
    device = "cuda"
    model = VAE(
        img_ch=3,
        base_ch=16,
        ch_mul=[2,4,8],
        attn=[],
        n_resblocks=2,
        e_ch=8
    ).to(device)
    
    image = torch.zeros(size=(1,3,64,64),device=device)
    
    recon, mean, std, encoding = model(image)
    print(recon.shape,encoding.shape)
    
    print(f"Param Count : {param_count(model)}")