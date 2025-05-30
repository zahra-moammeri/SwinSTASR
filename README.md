# SwinSTASR


In this study we have proposed two architectures **SwinWSR** and **SwinSTASR**.
Also we introduced a **transform attention block** in SwinSTASR.

## SwinWSR
<div style="text-align:center;">
    <img src="./assets/SwinWSR-model.png" alt= "SwinWSR Model" display='inline-block' width=50%>
</div>

## SwinSTASR
<div style="text-align:center;">
    <img src="./assets/SwinSTASR-model.png" alt= "SwinSTASR Model" display='inline-block' width=50%>
</div>

# Results

<!-- PSNR -->
* PSNR results at x2. The best results are in $${\color{red}red}$$ and the second bests are in $${\color{blue}blue}$$.

|   model   |  Set5    |   Set14   |   BSD100  |   Urban100    |   Manga109    |
|  :----:   |  :----:  |   :----:  |   :----:  |    :----:     |    :----:     |
|  SwinIR   | 38.2496  |  33.9940  | 32.3975   |  32.8472      |  39.4020      |
|  SwinFIR  | 38.3215  |  34.1799  |  32.4529  |   33.1972     |$${\color{blue}39.5940}$$|
|  SwinWSR  | $${\color{red}38.3589}$$ | $${\color{blue}34.2524}$$ | $${\color{blue}32.4542}$$ | $${\color{blue}33.2187}$$ | 39.5833 |
| SwinSTASR | $${\color{blue}38.3378}$$ | $${\color{red}34.2697}$$ | $${\color{red}32.4619}$$ | $${\color{red}33.2770}$$ | $${\color{red}39.6373}$$ |

<!-- SSIM -->
* SSIM results at x2. The best results are in $${\color{red}red}$$ and the second bests are in $${\color{blue}blue}$$.

|   model   |  Set5    |   Set14   |   BSD100  |   Urban100    |   Manga109    |
|  :----:   |  :----:  |   :----:  |   :----:  |    :----:     |    :----:     |
|  SwinIR   |  0.9616  |  0.9211   |  0.9024   |   0.9416      |   0.9787      |
|  SwinFIR  | $${\color{red}0.9623}$$| $${\color{blue}0.9227}$$| 0.9031 | 0.9493 | $${\color{blue}0.9790}$$|
|  SwinWSR  | $${\color{blue}0.9620}$$| 0.9226 | $${\color{blue}0.9033}$$| $${\color{blue}0.9495}$$| 0.9787| 
| SwinSTASR | 0.9619 | $${\color{red}0.9232}$$ | 0.9034 | $${\color{red}0.9501}$$ | $${\color{red}0.9792}$$|