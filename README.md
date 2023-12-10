# Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video
Yanqin Jiang<sup>1</sup>, [Li Zhang](https://lzrobots.github.io/)<sup>3</sup>, [Jin Gao](https://people.ucas.edu.cn/~jgao?language=en)<sup>1</sup>, [Weimin Hu](https://people.ucas.ac.cn/~huweiming?language=en)<sup>1</sup>, [Yao Yao](https://yoyo000.github.io/)<sup>2 ✉</sup> <br>
<sup>1</sup>CASIA, <sup>2</sup>Nanjin University, <sup>3</sup>Fudan University

| [Project Page](https://consistent4d.github.io/) | [arXiv](https://arxiv.org/abs/2311.02848) | [Paper](https://drive.google.com/file/d/1-6Ajm7AzAeAQ9IJLA1ntAUXJCZ0JOU56/view?usp=sharing) | Video (Coming soon) | [Data](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view?usp=sharing) |

![Demo GIF](https://github.com/consistent4d/consistent4d.github.io/blob/main/assets/demo.gif)

# Abstract
In this paper, we present Consistent4D, a novel approach for generating dynamic objects from uncalibrated monocular videos.

Uniquely, we cast the 360-degree dynamic object reconstruction as a 4D generation problem, eliminating the need for tedious multi-view data collection and camera calibration. 
This is achieved by leveraging the object-level 3D-aware image diffusion model as the primary supervision signal for training Dynamic Neural Radiance Fields (DyNeRF).
Specifically, we propose a Cascade DyNeRF to facilitate stable convergence and temporal continuity under the supervision signal which is discrete along the time axis. 
To achieve spatial and temporal consistency, we further introduce an Interpolation-driven Consistency Loss. 
It is optimized by minimizing the discrepancy between rendered frames from DyNeRF and interpolated frames from a pretrained video interpolation model.

Extensive experiments show that our Consistent4D can perform competitively to prior art alternatives, opening up new possibilities for 4D dynamic object generation from monocular videos, whilst also demonstrating advantage for conventional text-to-3D generation tasks. Our project page is [https://consistent4d.github.io/](https://consistent4d.github.io/)

# News
[**2023.12.10**] The code of Consistent4D is released! The code is refractored and optimized to accelerate training (**~2 hours** on a V100 GPU now!). For the convenience of quantitative comparison, we provide [**test dataset**](https://drive.google.com/file/d/1FwpP15k3fPGq8YG4s7bsogdJm6Gm86kg/view?usp=sharing) used in our paper and [our results](https://drive.google.com/file/d/1kyt78Er7ylO3hEz6Zn8ug31UqPa5kSNb/view?usp=sharing) on test dataset. <br>
[**2023.11.7**] The paper of Consistent4D is avaliable at [arXiv](https://arxiv.org/abs/2311.02848). We also provide input videos used in our paper [here](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view?usp=sharing). For our results on the input videos, please visit our [github project page](https://github.com/consistent4d/consistent4d.github.io) to download them (see folder `gallery`).

# Installation
**The installation is the same as the original threestudio, so skip it if you have already installed threestudio.** 
* You must have an NVIDIA graphics card with at least 24GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
* For video interpolation model, download **flownet.pkl** from [google drive](https://drive.google.com/file/d/1MWSTZYyRmm83asQQZBfPUJ9QsYohXSSy/view?usp=sharing), or use the latest version in [RIFE](https://github.com/hzwer/Practical-RIFE). We believe the latest version has better performace than old ones, but  we haven't tested it thoroughly. 

```bash
# Recommand to use annoconda
conda create -n consistent4d python=3.9
conda activate consistent4d
# Clone the repo
git clone https://github.com/yanqinJiang/Consistent4D
cd Consistent4D

# Build the environment
# Install torch: the code is tested with torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Install other packages
pip install -r requirement.txt


# Prepare Zero123
cd load/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt

# Prepare video interpolation module
cp /path/to/flownet/checkpoint ./extern/RIFE/flownet.pkl
```

# Data preparation
We provide the processed input data used in our paper. If you want to use your own data, please follow the steps below for pre-processing:
* Split the video to images, and name them after "{id}.png".
* Segment the foreground object in each frame. (We use [Tracking-Anything](https://github.com/gaomingqi/Track-Anything) to segment the foreground object.)
* Generate **RGBA images**. (It is a must!!)
* Sample the frames if you want. (For example, we sample 2s videos to 32 frames.)

The structure of the input data should be like
```bash
-image_seq_name
    - 0.png
    - 1.png
    - 2.png
    ...
```
# Training
```bash
# We provide three different configs (consistent4d_low_vram.yaml/consistent4d.yaml/consistent4d-4layers.yaml), requireing 24G/32G/40G VRAM for training, respectively. 
# The results in the paper and project page are produced by model in the config consistent4d.yaml. consistent4d_4layers.ymal is newly added, aiming at better results. 
# If you have aceess to GPU with enough memory, we higly recommand to set data.resolution_milestones in config to a larger number, i.e., 400, and you will get even better resutls.

python launch.py --config configs/consistent4d.yaml --train --gpu 0 data.image_seq_path=./load/demo/blooming_rose
```

# TODO
- [x] Release the code of DyNeRF training.
- [ ] Release the code of video enhancer.
- [ ] Provide evaluation scripts.
- [ ] Support advanced features (Maybe).

We have interest in continuously improving our work and add new features, i.e., advanced 4D representations and supervision signals. If you meet any problem during use or have any suggestions on improvement, please feel free to open an issue. We thanks for your feedback : ) !

# Tips
* **Multi-face(foot) Janus problem**: Multi-face(foot) Janus problem could be alleviate by the proposed Interpolation-based Consistency Loss (ICL). Increasing the vif_weight `system.loss.lambda_vif_{idx}` or the probabily of the loss `data.mode_section` in the config could amplify the effect of ICL. However, too large weight/probability usually results in over-smoothing. (The spatial and temporal data sample interval in ICL could also be modified by `data.random_camera.azimuth_interval_range` and `data.time_interval`, and increasing the sample interval intuitively wil amplify the effect of ICL too.)
* **Trade-off between VRAM and performance**: 
    * Usually, large batch size and rendering resolution (refer to  `data.batch_size` and `data.resolution_milestones` in config) when dyNeRF training could lead to better results. However, it requires large GPU memory.
    * Increasing cascade dynamic NeRF layers and plane resolution in `system.geometry.grid_size` also leads to better performance. (But **do not** make the temporal resolution, i.e., the last number in grid_size, larger than the number of frames). The former costs more GPU memeory but **the latter not**.
* **Different seed**: We haven't tested the effect of different seed in video-to-4d task (our experiments are all run with seed 0), however, we believe sometimes a simple solution to multi-face Janus problem/bad texture is to try a different seed. Good luck!
* **Better texture in input view**: If you want to get better texture in input view, you could fintune the model with increased rgb loss and decreased sds loss after inital training. However, this could only improve the texture in input view, leaving texture for invisible views almost unchanged.

# Acknowledgement 
Our code is based on [Threestudio](https://github.com/threestudio-project/threestudio). We thank the authors for their effort in building such a great codebase. <br>
The video interpolation model employed in our work is [RIFE](https://github.com/hzwer/Practical-RIFE), which is continuously improved by its authors for real-world application. Thanks for their great work!
# Citation
```bibtex
 @article{jiang2023consistent4d,
     author = {Jiang, Yanqin and Zhang, Li and Gao, Jin and Hu, Weimin and Yao, Yao},
     title = {Consistent4D: Consistent 360 $\{$$\backslash$deg$\}$ Dynamic Object Generation from Monocular Video},
     journal = {arXiv preprint arXiv:2311.02848},
     year = {2023},
 }
```


