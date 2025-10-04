## SWAB

import torch
import torch.nn as nn
import pytorch_wavelets as pw


class ResBlock(nn.Module):
    """Calculate Residual Block. Spatial part"""
    def __init__(self, embed_dim):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope= 0.2, inplace= True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )

    def forward(self, x):
        out = self.body(x)
        return out + x
    

class WaveletUnit(nn.Module):
    """Calculate wavelet unit"""
    def __init__(self, embed_dim, J = 4, wave = 'db4', mode = 'symmetric'):
        super(WaveletUnit, self).__init__()

        self.embed_dim = embed_dim
        self.J = J
        self.wave = wave
        self.mode = mode

        self.LL_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim , 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim * 3, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # -------------------- Added -------------------
        self.avg_pool_seq_LL = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(embed_dim , embed_dim , 1, 1, 0)       
            )  # added
        
        self.avg_pool_seq_subs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(embed_dim , embed_dim * 3, 1, 1, 0),
            )  # added
        # ---------------------------------------------

        
        self.dwt = pw.DWTForward(J=self.J, wave=self.wave, mode=self.mode)
        self.inv_dwt = pw.DWTInverse(wave=self.wave, mode=self.mode)
        
    def forward(self, x):
        
        input = x
        x = x
        # ---------------Added---------------
        avg_pool_LL = self.avg_pool_seq_LL(input)
        avg_pool_subs = self.avg_pool_seq_subs(input) 
        # ------------------------------------------

        cA, subbands = self.dwt(x)
        cA_conv = self.LL_conv(cA)    # [4, 60, 10, 10]
        
        cA_conv = cA_conv * avg_pool_LL # ADDED
        sub_list = []

        for i, _ in enumerate(subbands):
            subband = subbands[i]                # subbannds: [4, 60, 3, 33, 33] ,  [4, 60, 3, 20, 20],  [4, 60, 3, 13, 13] 
            b, c, k, h, w = subband.shape
            subband = subband.reshape(b, -1 , h, w)  # [4, 60, 3, h, w] --> [4, 180, h, w]
            subband_conv = self.conv_layer(subband)      # [4, 180, h, w]

            subband_conv = subband_conv * avg_pool_subs   # ADDED
            subband = subband_conv.view(b, c, k, h, w)     # [4, 180, h, w]--> [4, 60, 3, h, w]      
            sub_list.append(subband)

        out= self.inv_dwt((cA_conv, sub_list))    # [4, 60, 60, 60]
        return out
    

class WaveTransform(nn.Module):
    def __init__(self, embed_dim, wave = 'db4', mode = 'symmetric', *args, **kwars):
        super(WaveTransform, self).__init__()

        self.J = 1
        self.wave = wave
        self.mode = mode
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 3, 1, 1, 0),     # [4,  180, 60, 60]  --> [4, 60, 60, 60]
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.wave = WaveletUnit(embed_dim // 3, J = self.J, wave= self.wave, mode=self.mode)  # [4, 60, 60, 60]
        self.conv2 = nn.Conv2d(embed_dim // 3, embed_dim, 1, 1, 0)   # [4, 60, 60, 60] --> [4, 180, 60, 60]


    def forward(self, x):
        x = self.conv1(x)          # [4, 180, 60, 60] -> [4, 60, 60, 60]
        wave = self.wave(x)    # [4, 60, 60, 60]
        output = self.conv2(wave + x)   
        return output  
    

class SWAB(nn.Module):
    def __init__(self, embed_dim):
        super(SWAB, self).__init__()
        self.res = ResBlock(embed_dim)
        self.wave = WaveTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x):
        res = self.res(x)
        wave = self.wave(x)
        out = torch.cat([res, wave], dim=1)
        out = self.fusion(out)

        return out