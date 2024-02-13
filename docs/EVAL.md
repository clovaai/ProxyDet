
## Open-Vocab LVIS evaluation

- for evaluation of our proxydet models, download model first and execute evaluation script referred to below.

|         Name          | Backbone | Training dataset |  mask mAP | mask mAP_novel  | Download | Evaluation
|-----------------------|-----------------------|------------------|-----------|-----------------|----------|----------|
|[BoxSup-R50](../configs/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.yaml)     | ResNet50 | LVIS | 30.2      |       16.4      | [model](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth) | -
|[ProxyDet-R50 (wo/ inl)](../configs/ProxyDet_R50_Lbase_INL.yaml)     | ResNet50 | LVIS | 30.1      |       19.0 (+2.6)      | [model](https://drive.google.com/file/d/1H8cFzJRht8o9yH2eIyopfQo09UgF-5ij/view?usp=sharing) | [script](../scripts/eval/proxydet_r50_wo_inl.sh)
| | || | | ||
|[Detic-R50](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml)    | ResNet50 | LVIS + IN-L | 32.4      |   24.9      | [model](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth) | -
|[ProxyDet-R50 (w/ inl) ](../configs/ProxyDet_R50_Lbase_INL.yaml)     | ResNet50 | LVIS + IN-L | 32.8      |   26.2 (+1.3)      | [model](https://drive.google.com/file/d/1IEfoSPRGYWtaxk9sKhBf3NSBj6NWGgvt/view?usp=sharing) | [script](../scripts/eval/proxydet_r50_w_inl.sh)
| | || | | ||
|[Detic-SWINB](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml)    | SWIN-B | LVIS + IN-L | 40.7      |   33.8      | [model](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) | -
|[ProxyDet-SWINB (w/ inl) ](../configs/ProxyDet_SwinB_Lbase_INL.yaml)     | SWIN-B | LVIS + IN-L | 41.5      |   36.7 (+2.9)      | [model](https://drive.google.com/file/d/17kUPoi-pEK7BlTBheGzWxe_DXJlg28qF/view?usp=sharing) | [script](../scripts/eval/proxydet_swinb_w_inl.sh)


- for evaluation on non-pseudo-labeled novel classes, run:

```
LVIS_INSTASNCE_RESULT_FILE_PATH="YOUR_${LVIS_INSTASNCE_RESULT_FILE_PATH}"

cd tools && python category_wise_ap_lvis.py ${LVIS_INSTASNCE_RESULT_FILE_PATH}
```

- AP result of ```ProxyDet-R50 (w/ inl)``` on pseudo-labeled novel classes / non-pseudo-labeled novel classes / all novel classes
```
frequency_group: rare, category_group: in_im
ap: 26.976
frequency_group: rare, category_group: not_in_im
ap: 22.998
frequency_group: rare, category_group: all
ap: 26.216
```