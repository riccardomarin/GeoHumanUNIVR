# GeoHumanUNIVR
Repository for the short course: Geometric Deep Learning for Virtual Humans, UNIVR, November 2025

# Day One - Geometric Deep Learning
### 🛠️ Prerequisites & Demos

* **MeshLab:** A powerful tool for processing 3D meshes.
    * **Download & Install:** [MeshLab Website](https://www.meshlab.net/#download)
    * **Install-free Javascript Version:** [MeshLabJS](https://www.meshlabjs.net/)
* **Hunyuan3D Demo:** Explore 3D generation in this online space.
    * [Hugging Face Space](https://huggingface.co/spaces/tencent/Hunyuan3D-2)
* **Luma AI:** Interactive scenes and captures.
    * [Interactive Scenes](https://lumalabs.ai/interactive-scenes)
    * [Captures Dashboard](https://lumalabs.ai/dashboard/captures)

### 📓 Course Notebooks (Colab)

1.  **Visualize and Process:**
    * [Colab Link](https://colab.research.google.com/drive/1-52YibyE4C0f9NJP3o1MUMCxohaUCKh0?usp=sharing)
2.  **Learning:**
    * [Colab Link](https://colab.research.google.com/drive/1eZWc55Nti40DOu-zqRsGVL49AOs-C6m-?usp=sharing)
    
# Day Two - Virtual Humans
---
### 💻 Code & Environment Setup

You need to set up a local environment (a GPU is **not required**).

```
conda create -n sav3d python=3.9
conda activate sav3d

conda install -c conda-forge libstdcxx-ng
pip install smplx[all] open3d plyfile moderngl-window==2.4.6 pyglet aitviewer robust_laplacian scikit-learn pandas

pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
```
### 🔑 Required Accounts & Data
You need to create accounts on the following platforms to download necessary models and data:
- SMPLfy: https://smplify.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/
 
**Data Fetching:**
After creating your accounts, run the scripts in the **`scripts`** folder to fetch the data needed for the course scripts. Both Linux Bash and Windows PowerShell versions are available.
In case of troubles, just perform the instructions manually.

----

## 📚 References & Further Reading

This tutorial takes inspiration from a number of sources that are useful for diving deeper into the topics. If you reuse some of the material from this tutorial, please give them proper credit:

* **SMPL Made Simple Tutorial:** [smpl-made-simple.is.tue.mpg.de](https://smpl-made-simple.is.tue.mpg.de/)
* **"Virtual Humans" Lecture (University of Tuebingen):** [YouTube Playlist](https://www.youtube.com/watch?v=DFHuV7nOgsI&list=PL05umP7R6ij13it8Rptqo7lycHozvzCJn)
* **Meshcapade Wiki:** [meshcapade.wiki](https://meshcapade.wiki/)
* **FAUST Dataset:** [faust-leaderboard.is.tuebingen.mpg.de](https://faust-leaderboard.is.tuebingen.mpg.de/)
