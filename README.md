# Web 3DGS Viewer

Real-time [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) in the browser, built on WebGPU. PLY scenes are streamed on the fly — no install, no native runtime.

**[Try it live →](https://saxenabhishek.me/splats/)**

## Performance

| Optimization | Result |
|---|---|
| WebGPU worker depth sort | 2x faster than CPU radix at scale |
| OBB frustum culling + tile pruning | 1.2x rendering speedup in large scenes |
| Sustained framerate | 30 FPS past 2M splats |

## Stack

WebGPU, WGSL compute shaders, JavaScript
