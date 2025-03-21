Uni-3DAR
========
[[Paper](https://arxiv.org/pdf/2503.16278)]

Introduction
------------

<p align="center"><img src="fig/overview.png" width=80%></p>
<p align="center"><b>Schematic illustration of the Uni-3DAR framework</b></p>

Uni-3DAR is an autoregressive model that unifies various 3D tasks. In particular, it offers the following improvements:

1. **Unified Handling of Multiple 3D Data Types.**  
   Although we currently focus on microscopic structures such as molecules, proteins, and crystals, the proposed method can be seamlessly applied to macroscopic 3D structures.

2. **Support for Diverse Tasks.**  
   Uni-3DAR naturally supports a wide range of tasks within a single model, especially for both generation and understanding.

3. **High Efficiency.**  
   It uses octree compression-in combination with our proposed 2-level subtree compression-to represent the full 3D space using only hundreds of tokens, compared with tens of thousands in a full-size grid. Our inference benchmarks also show that Uni-3DAR is much faster than diffusion-based models.

4. **High Accuracy.**  
   Building on octree compression, Uni-3DAR further tokenizes fine-grained 3D patches to maintain structural details, achieving substantially better generation quality than previous diffusion-based models.


News
----

**2025-03-21:** We have released the core model along with the QM9 training and inference pipeline.


Dependencies
------------

- [Uni-Core](https://github.com/dptech-corp/Uni-Core). For convenience, you can use our prebuilt Docker image:  
  `docker pull dptechnology/unicore:2407-pytorch2.4.0-cuda12.5-rdma`






