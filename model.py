import torch
import torch.nn as nn
from einops import repeat
import math
from torch.nn import init

class TimeEmbeddings(nn.Module):
    def __init__(self, steps, d_model, time_dim):
        super().__init__()
        
        assert time_dim % 2 == 0
        '''
        [pos  * [1/10000^index0/d_model , 1/10000^index1/d_model, 1/10000^index2/d_model]
         pos,
         pos]
         
        [ pos/10000^index0/d_model , pos/10000^index1/d_model, pos/10000^index2/d_model]
        [ pos/10000^index0/d_model , pos/10000^index1/d_model, pos/10000^index2/d_model]
        [ pos/10000^index0/d_model , pos/10000^index1/d_model, pos/10000^index2/d_model]
        
        now sin and cos the above matrix
        Stack and reshape

        '''
        
        pos = torch.arange(0,steps)
        
        emb = ( torch.arange(0,d_model,2) / d_model ) * math.log(10000)
        emb = torch.exp(-emb)
        emb = pos[:,None] * emb[None,:]
        
        emb = torch.stack([torch.sin(emb),torch.cos(emb)], dim=-1)
        emb = emb.view(steps, d_model)
        
        self.register_buffer("emb", emb)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim)
        )
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, t):
        if t.dim() == 2:
            t = t.squeeze(1)
        emb = self.emb[t]           # (B, d_model)
        return self.time_mlp(emb)   # (B, time_dim)
    
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)
        self.initialize()
        
    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
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

    def forward(self, x, temb):
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
    def __init__(self, in_channels, out_channels, attn, tdim):
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
        
        self.tdim = tdim
        self.time_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim,out_channels)
        )
    
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x, temb):
                
        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)
        
        t_emb = self.time_linear(temb)
        out = out + t_emb[:,:,None,None]

        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        out = out + self.shortcut(x)
        out = self.attn(out)
        return out

class UNet(nn.Module):
    def __init__(self, img_ch, base_ch, ch_mul, attn, n_resblocks, steps, tdim):
        super().__init__()
        
        self.time_embedding = TimeEmbeddings(steps, base_ch, tdim)
        
        self.img_ch = img_ch
        self.initial_conv = nn.Conv2d(img_ch, base_ch, 3 ,1, 1)
        
        channels_list = [base_ch]
        current_ch = base_ch
        self.encoder_blocks = nn.ModuleList()
        for i,mul in enumerate(ch_mul):
            out_ch = base_ch * mul
            for _ in range(n_resblocks):
                self.encoder_blocks.append(
                    ResBlock(current_ch,out_ch,attn=(i in attn),tdim=tdim)
                    )
                current_ch = out_ch
                channels_list.append(out_ch)
            if i != len(ch_mul) - 1:
                self.encoder_blocks.append(
                    Downsample(current_ch,out_ch)
                )
                current_ch = out_ch
                channels_list.append(out_ch)
        
        self.middleblocks = nn.ModuleList([
            ResBlock(current_ch, current_ch, tdim=tdim, attn=True),
            ResBlock(current_ch, current_ch, tdim=tdim, attn=False),
        ])
        
        self.decoder_blocks = nn.ModuleList()
        for i,mul in reversed(list(enumerate(ch_mul))):
            out_ch = base_ch * mul
            for _ in range(n_resblocks + 1):
                self.decoder_blocks.append(
                    ResBlock(channels_list.pop() + current_ch,out_ch,attn=(i in attn),tdim=tdim)
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
    
    def forward(self, x, t):
        
        temb = self.time_embedding(t)
        
        x = self.initial_conv(x)
        
        features = [x]
        for layer in self.encoder_blocks:
            x = layer(x,temb)
            features.append(x)
        
        for layer in self.middleblocks:
            x = layer(x, temb)

        for layer in self.decoder_blocks:
            if isinstance(layer,ResBlock):
                x = torch.concat([x,features.pop()],dim=1)
            x = layer(x,temb)
        
        x = self.proj_out(x)
        return x

def param_count(model):
    param_count = 0
    for params in model.parameters():
        param_count += params.numel()
    return param_count

if __name__=="__main__":
    device = "cuda"
    model = UNet(
        img_ch=8,
        base_ch=128,
        ch_mul=[1,2,2],
        attn=[1,2],
        n_resblocks=2,
        steps=1000,
        tdim=512
    ).to(device)
    
    batch_size = 2
    image = torch.zeros(size=(batch_size,8,16,16),device=device)
    time = torch.full((batch_size,),0,dtype=torch.long,device=device )
    
    noise = model(image,time)
    print(noise.shape)
    
    print(f"Param Count : {param_count(model)}")