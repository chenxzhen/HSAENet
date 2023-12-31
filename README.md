# HSAENet (Hybrid Spatial-Angular Exploitation Network)
We introduce a novel Hybrid Spatial-Angular Exploitation (HSAE) module to exploit spatial-angular information from multiple light field image representations and then integrate the efficient incorporation of the most relevant information into a comprehensive feature representation. While accounting for the interrelation and disparity among these representations, we propose an adaptive fusion strategy, Iterative Hybrid Feature Fusion (IHFF), allowing for the selective and context-aware integration of diverse feature representations. Additionally, we present an innovative aggregation technique that harnesses both shallow and deep feature aggregation, facilitating the capture of hierarchical contextual information. Based on the above modules, we construct a Hybrid Spatial Angular Exploitation Network termed as **HSAENet**. Compared with the current leading approach EPIT, the proposed HSAENet achieves PSNR improvements of **0.20dB** and **0.41dB** on the performance of average values across five datasets for **x2** and **x4**.

# Schedule
- [ ] Release code
- [ ] Release x2 and x4 models

# Overview

![image](image/overview.png)

# Quantitative Results
Please see training log ([HSAENet_x4_log](HSAENet_x4_log.txt) [HSAENet_x2_log](HSAENet_x2_log.txt)) on five commonly-used datasets (*EPFL*, *HCI_new*, *HCI_old*, *INRIA_Lytro*, *Stanford_Gantry*).

![image](image/Quantitative_Results.jpg)


