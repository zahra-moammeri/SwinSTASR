# SwinSTASR


In this study we have proposed two architectures **SwinWSR** and **SwinSTASR**.
Also we introduced a **transform attention block** in SwinSTASR.

## SwinWSR
<div style="display:flex; justify-content:center;">
    <img src="./assets/SwinWSR-model.png" alt= "SwinWSR Model" width=70%>
</div>

## SwinSTASR
<div style="display:flex; justify-content:center;">
    <img src="./assets/SwinSTASR-model.png" alt= "SwinSTASR Model" width=70%>
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

<hr/>

<!-- SSIM -->
* SSIM results at x2. The best results are in $${\color{red}red}$$ and the second bests are in $${\color{blue}blue}$$.

|   model   |  Set5    |   Set14   |   BSD100  |   Urban100    |   Manga109    |
|  :----:   |  :----:  |   :----:  |   :----:  |    :----:     |    :----:     |
|  SwinIR   |  0.9616  |  0.9211   |  0.9024   |   0.9416      |   0.9787      |
|  SwinFIR  | $${\color{red}0.9623}$$| $${\color{blue}0.9227}$$| 0.9031 | 0.9493 | $${\color{blue}0.9790}$$|
|  SwinWSR  | $${\color{blue}0.9620}$$| 0.9226 | $${\color{blue}0.9033}$$| $${\color{blue}0.9495}$$| 0.9787| 
| SwinSTASR | 0.9619 | $${\color{red}0.9232}$$ | 0.9034 | $${\color{red}0.9501}$$ | $${\color{red}0.9792}$$|


<hr/>

<!-- LPIPS -->
* LPIPS results at x2. The best results are in $${\color{red}red}$$ and the second bests are in $${\color{blue}blue}$$.

|   model   |  Set5    |   Set14   |   BSD100  |   Urban100    |   Manga109    |
|  :----:   |  :----:  |   :----:  |   :----:  |    :----:     |    :----:     |
|  SwinIR   |  0.0525  |  0.0848   |   0.1206  |    0.0349     |    0.0220     |
|  SwinFIR  | $${\color{blue}0.0514}$$ | 0.0840 | 0.1193 |  0.0321 | $${\color{red}0.0216}$$ |
|  SwinWSR  | $${\color{blue}0.0514}$$ | $${\color{blue}0.0833}$$ | $${\color{red}0.1169}$$ | $${\color{red}0.0314}$$ | $${\color{blue}0.0217}$$ |
| SwinSTASR | $${\color{red}0.0513}$$ | $${\color{red}0.0830}$$ | $${\color{blue}0.1178}$$ | $${\color{blue}0.0319}$$ | $${\color{red}0.0216}$$ |


<hr/>

<!-- LPIPS -->
* SwinSTASR results at x4.

|   Metric  |  Set5    |   Set14   |   BSD100  |   Urban100    |   Manga109    |
|  :----:   |  :----:  |   :----:  |   :----:  |    :----:     |    :----:     |
|  PSNR     | 32.8065  | 29.0439   |  27.8641  |    27.3730    |   31.9224     |
|  SSIM     | 0.9023   | 0.7928    |  0.7471   |     0.8226    |    0.9245     |
|  LPIPS    |   0.1707 |   0.2711  |   0.3289  |    0.1832     |    0.1018     |

## Compare models at x2
<div style="display:flex; justify-content:center;">
    <img src="./assets/Swinfir-stasr-swb-compares.png" alt= "Compare Models" width=90%>
</div>

<hr/>

<div style="display:flex; justify-content:center;">
    <img src="./assets/Urban_img033_compares.png" alt= "Compare Models" width=90%>
</div>

<hr/>

<div style="display:flex; justify-content:center;">
    <img src="./assets/Urban_img062_compares.png" alt= "Compare Models" width=90%>
</div>

