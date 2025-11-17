# GeoHumanUNIVR
Repository for the short course: Geometric Deep Learning for Virtual Humans, UNIVR, November 2025

# Day One

- You may want to download and install Meshlab:
Download: https://www.meshlab.net/#download
or you can use the (install-free) Javascript version: https://www.meshlabjs.net/

- Hunyan3D demo: https://huggingface.co/spaces/tencent/Hunyuan3D-2
- LumaAI
https://lumalabs.ai/interactive-scenes \
https://lumalabs.ai/dashboard/captures

## Notebooks
1) Visualize and Process:\
   https://colab.research.google.com/github/riccardomarin/CSW25Geo3D/blob/main/CSW25_VisualizeAndProcess.ipynb

2) Learning:\
   https://colab.research.google.com/github/riccardomarin/CSW25Geo3D/blob/main/CSW_Learning.ipynb


# Day Two
---
You need to setup a local environment (without GPU is fine)

## 💻 Code & Environment Setup
```
conda create -n sav3d python=3.9
conda activate sav3d

conda install -c conda-forge libstdcxx-ng
pip install smplx[all] open3d plyfile moderngl-window==2.4.6 pyglet aitviewer robust_laplacian scikit-learn pandas

pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
```
You also need to create an account on:

- SMPLfy: https://smplify.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/

Then, run the scripts in the `scripts` folder to fetch the data needed to run the scripts. Both Linux Bash and Windows PowerShell versions of the scripts are available.
In case of trouble, just perform the instructions manually.

----

This tutorial takes inspiration from a number of sources, useful for diving deeper into the topics. In case you reuse some of the material from this tutorial, please give them proper credit:

- SMPL made simple tutorial: https://smpl-made-simple.is.tue.mpg.de/
- "Virtual Humans" Lecture from University of Tuebingen: https://www.youtube.com/watch?v=DFHuV7nOgsI&list=PL05umP7R6ij13it8Rptqo7lycHozvzCJn
- Meshcapade Wiki: https://meshcapade.wiki/
- FAUST dataset: https://faust-leaderboard.is.tuebingen.mpg.de/
