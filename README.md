# BiFormer

BiFormer: Bilateral Interaction for Local-Global Collaborative Perception in Low-Light Image Enhancement,&#x20;

Published in: *IEEE Transactions on Multimedia (TMM), 2024*

[paper](10.1109/TMM.2024.3413293) | [datasets](https://pan.baidu.com/s/12g91-HIxCdq36DV-Pt8wvQ?pwd=ssfh) | [results](https://pan.baidu.com/s/12g91-HIxCdq36DV-Pt8wvQ?pwd=ssfh)

## Abstract

Low-light image enhancement is a challenging task due to the limited visibility in dark environments. While recent advances have shown progress in integrating CNNs and Transformers, the inadequate local-global perceptual interactions still impedes their application in complex degradation scenarios. To tackle this issue, we propose BiFormer, a lightweight framework that facilitates local-global collaborative perception via bilatera linteraction. Specifically, our framework introduces a core CNN-Transformer collaborative perception block (CPB) that combines local-aware convolutional attention (LCA) and global-aware recursive Transformer (GRT) to simultaneously preserve local details and ensure global consistency. To promote perceptual interaction, we adopt bilateral interaction strategy for both local and global perception, which involves local-to-global second-order interaction (SoI) in the dual-domain, as well as a mixed-channel fusion (MCF) module for global-to-local interaction. The MCF is also a highly efficient feature fusion module tailored for degraded features. Extensive experiments conducted on low-level and high-level tasks demonstrate that BiFormer achieves state-of-the-art performance. Furthermore, it exhibits a significant reduction in model parameters and computational cost compared to existing Transformer-based low-light image enhancement methods.

## Overview

## ![](README_md_files/d1df7540-aa6b-11ef-90fb-6d0a30377dbd.jpeg?v=1\&type=image)

## Main Results

The enhanced results for the datasets mentioned in our paper can be downloaded through [Baidu Netdisk](https://pan.baidu.com/s/12g91-HIxCdq36DV-Pt8wvQ?pwd=ssfh) (code: ssfh) or  [Google Drive](https://drive.google.com/drive/folders/1g_LD_NHYz37jvM4T-RicTAQOeNMuFyP5?usp=sharing). Results are measured using `measure_pair.py` and `measure_nonpair.py`, respectively. It should be noted that all metrics in our method are computed in the sRGB space, and no GT Mean-related techniques are applied. 

![](README_md_files/cc9da200-aa6b-11ef-90fb-6d0a30377dbd.jpeg?v=1\&type=image)

![](README_md_files/c11ab3f0-aa6b-11ef-90fb-6d0a30377dbd.jpeg?v=1\&type=image)

![](README_md_files/c52cb9c0-aa6b-11ef-90fb-6d0a30377dbd.jpeg?v=1\&type=image)

## Citation

If you find this work useful for your research, please cite:

    @ARTICLE{BiFormer,
      author={Xu, Rui and Li, Yuezhou and Niu, Yuzhen and Xu, Huangbiao and Chen, Yuzhong and Zhao, Tiesong},
      journal={IEEE Transactions on Multimedia}, 
      title={Bilateral Interaction for Local-Global Collaborative Perception in Low-Light Image Enhancement}, 
      year={2024},
      volume={26},
      number={},
      pages={10792-10804},
      keywords={Convolutional neural networks;Transformers;Image enhancement;Task analysis;Collaboration;Visualization;Lighting;Low-light image enhancement;hybrid CNN-Transformer;bilateral interaction;mixed-channel fusion},
      doi={10.1109/TMM.2024.3413293}}

