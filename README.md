# Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video
Yanqin Jiang<sup>1</sup>, [Li Zhang](https://lzrobots.github.io/)<sup>3</sup>, [Jin Gao](https://people.ucas.edu.cn/~jgao?language=en)<sup>1</sup>, [Weimin Hu](https://people.ucas.ac.cn/~huweiming?language=en)<sup>1</sup>, [Yao Yao](https://yoyo000.github.io/)<sup>2 ✉</sup> <br>
<sup>1</sup>CASIA, <sup>2</sup>Nanjin University, <sup>3</sup>Fudan University

| [Project Page](https://consistent4d.github.io/) | arXiv | [Paper](https://drive.google.com/file/d/1_WUKUwZBUTybayZ81mzQYPIo9Djq7kPb/view?usp=sharing) | Video (Coming soon) | [Data](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view?usp=sharing) |

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
[**2023.11.6**] The paper of Consistent4D is avaliable at arXiv. We also provide input videos used in our paper [here](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view?usp=sharing).  

# Citation
```bibtex
 @article{jiang2023consistent4d,
     author = {Jiang, Yanqin and Zhang, Li and Gao, Jin and Hu, Weimin and Yao, Yao},
     title = {Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video},
     journal = {arxiv},
     year = {2023},
 }
```


