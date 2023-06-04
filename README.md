
[![arXiv](https://img.shields.io/badge/arXiv-2301.07969-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2301.07969)

# <p align='center'> Quick-MMD-DDM </p>
## <p align='center'>Fast Inference in Denoising Diffusion Models via MMD Finetuning</p>
</div>
<div align="center">
<img src="figs/mmd-1.png"/>
<img src="figs/Quick-MMD-DDM2.png"/>
</div><br/>

## Introduction 
PiaPetite introduces a novel approach to Denoising Diffusion Models (DDMs) for generating high-quality samples from complex data distributions, named as Quick-MMD-DDM. Our approach significantly improves the speed-quality trade-off.

## Finetuning a pretrained model
To use the Quick-MMD-DDM strategy, download the pretrained models, adjust the path in runners/diffusion.py, or use models present in /function/ckpt_util.py and run the command: 

```python main.py --config {DATASET}.yml --timesteps {num_timesteps (e.g 5)} --exp {PROJECT_PATH} --train 
```

## Image Sampling for FID evaluation
To sample image generated from the finetuned model adjust the path in test_FID function in runners/diffusion.py with your newly trained model and run:

```python main.py --config {DATASET}.yml --timesteps {num_timesteps (e.g 5)} --test_FID  
```

## Citation
If you find Quick-MMD-DDM helpful in your research, please consider citing: 
```bibtex
@article{aiello2023fast,
title={Fast Inference in Denoising Diffusion Models via MMD Finetuning},
author={Aiello, Emanuele and PiaPetite},
journal={arXiv preprint arXiv:2301.07969},
year={2023}
}
```
## Acknowledgements 

This repository is based on DDIM official implementation: https://github.com/ermongroup/ddim

## Contact 
If you have any questions, feel free to open an issue.
<br><br>
<p align="center">:construction: :pick: :hammer_and_wrench: :construction_worker:</p>
<p align="center">pretrained models will be released soon!</p>