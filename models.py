import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalWaveNet(nn.Module):
    def __init__(self, in_channels=256, res_channels=64, out_channels=256, num_blocks=2, num_layers=4):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                self.blocks.append(
                    nn.Conv1d(res_channels, res_channels, kernel_size=2, dilation=dilation, padding=dilation)
                )
        self.output1 = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.output2 = nn.Conv1d(res_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x)
        for conv in self.blocks:
            res = x
            x = torch.relu(conv(x))
            x = x[..., :res.shape[-1]] 
            x = x + res
        x = F.relu(self.output1(x))
        x = self.output2(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, cond_dim):
        super().__init__()
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.condition_filter = nn.Conv1d(cond_dim, channels, kernel_size=1)
        self.condition_gate = nn.Conv1d(cond_dim, channels, kernel_size=1)
        self.output_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, cond):
        f_x = self.filter_conv(x)
        g_x = self.gate_conv(x)
    
        cond_interp = F.interpolate(cond, size=f_x.shape[-1])
        
        # Add conditioning
        f = f_x + self.condition_filter(cond_interp)
        g = g_x + self.condition_gate(cond_interp)
    
        z = torch.tanh(f) * torch.sigmoid(g)
        out = self.output_conv(z)
    
        if out.shape[-1] != x.shape[-1]:
            diff = x.shape[-1] - out.shape[-1]
            out = F.pad(out, (0, diff))  # Pad at the end to match
    
        return x + out


class ConditionalWaveNet(nn.Module):
    def __init__(self, 
                 n_classes=256, 
                 cond_dim=64, 
                 res_channels=64, 
                 num_blocks=2, 
                 num_layers=6, 
                 num_instruments=128, 
                 num_pitches=128):
        super().__init__()

        # Conditioning embeddings
        self.inst_embed = nn.Embedding(num_instruments, cond_dim // 2)
        self.pitch_embed = nn.Embedding(num_pitches, cond_dim // 2)

        self.input_conv = nn.Conv1d(n_classes, res_channels, kernel_size=1)
        # self.blocks = nn.ModuleList([
        #     ResidualBlock(res_channels, dilation=2**i, cond_dim=cond_dim)
        #     for _ in range(num_blocks)
        #     for i in range(num_layers)])
        dilation_cycle = [1, 2, 4, 8] 
        self.blocks = nn.ModuleList([
            ResidualBlock(res_channels, dilation=d, cond_dim=res_channels)
            for _ in range(num_blocks)
            for d in dilation_cycle])
        self.output1 = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.output2 = nn.Conv1d(res_channels, n_classes, kernel_size=1)

    def forward(self, x, inst, pitch):
        # Embed and concatenate condition inputs
        cond = torch.cat([
            self.inst_embed(inst),  # (B, cond_dim//2)
            self.pitch_embed(pitch)  # (B, cond_dim//2)
        ], dim=1).unsqueeze(-1)  # (B, cond_dim, 1)

        x = self.input_conv(x)  # (B, res_channels, T)
        for block in self.blocks:
            x = block(x, cond)   # residual inside
        x = F.relu(self.output1(x))
        return self.output2(x)  # (B, n_classes, T)
