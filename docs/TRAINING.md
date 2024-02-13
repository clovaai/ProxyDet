## Preliminary


### Pretrained model download
- ResNet-50 backbone: download LVIS-based pretrained BoxSup model checkpoint
```bash
wget https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth -O models/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth
```
- Swin-B backbone: download LVIS-based pretrained BoxSup model checkpoint
```bash
wget https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.pth -O models/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.pth
```

### Download training datasets
- download LVIS and COCO datasets, following [here](https://github.com/facebookresearch/Detic/tree/main/datasets#coco-and-lvis)
- download ImageNet-LVIS, following [here](https://github.com/facebookresearch/Detic/tree/main/datasets#imagenet-21k)
- place all the datasets into ```datasets``` directory.

### Reproducibility

- for reproducibility, you should modify detectron2's sources by:
- for more details, please see [here](https://github.com/facebookresearch/detectron2/issues/4723)

```python
# detectron2/layers/roi_align.py#L58
return roi_align(
    input.half().double(),
    rois.half().double(),
    self.output_size,
    self.spatial_scale,
    self.sampling_ratio,
    self.aligned,
).to(dtype=input.dtype)
```

<details>
  <summary>training environment</summary>
  
  ```
----------------------  -----------------------------------------------------------------------
sys.platform            linux
Python                  3.8.10 (default, Jun 22 2022, 20:18:18) [GCC 9.4.0]
numpy                   1.23.2
detectron2              0.6
Compiler                GCC 9.4
CUDA compiler           CUDA 11.3
detectron2 arch flags   7.0
DETECTRON2_ENV_MODULE   <not set>
PyTorch                 1.11.0+cu113
PyTorch debug build     False
GPU available           Yes
GPU 0,1,2,3,4,5,6,7     Tesla V100-SXM2-32GB (arch=7.0)
Driver version          515.48.07
CUDA_HOME               /usr/local/cuda
Pillow                  9.2.0
torchvision             0.12.0+cu113
torchvision arch flags  3.5, 5.0, 6.0, 7.0, 7.5, 8.0, 8.6
fvcore                  0.1.5.post20220512
iopath                  0.1.9
cv2                     4.6.0
----------------------  -----------------------------------------------------------------------
PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.5.2 (Git Hash a9302535553c73243c632ad3c4c80beec3d19a1e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 
```
  
</details>

- Note: training results might vary depending on the GPU devices & driver versions.

# training

- to train our proxydet (ResNet-50 backbone) on LVIS + ImageNet-LVIS datasets, run

```
bash scripts/train/proxydet_r50_w_inl.sh
```

- [training logs](https://drive.google.com/file/d/1CjYjT25XEsCwxZ8esHOQccWZAtpt8Iak/view?usp=sharing)

- for training Swin-B backbone, run

```
bash scripts/train/proxydet_swinb_w_inl.sh
```

- [training logs](https://drive.google.com/file/d/19G1a3n4tvPKnjUZ03R0UQ0E2OiP3YGPn/view?usp=sharing)


