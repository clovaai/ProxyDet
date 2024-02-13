# Copyright (c) Facebook, Inc. and its affiliates.
'''
    Modifications Copyright (c) 2024-present NAVER Corp, Apache License v2.0
    original source: https://github.com/facebookresearch/Detic/blob/main/detic/modeling/roi_heads/zero_shot_classifier.py
'''
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        use_regional_embedding: bool = False
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

        self.use_regional_embedding = use_regional_embedding
        if self.use_regional_embedding:
            assert (
                self.norm_weight
            ), "norm_weight should be True for using regional embedding."


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            "use_regional_embedding": cfg.MODEL.ROI_BOX_HEAD.USE_REGIONAL_EMBEDDING,
        }

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            # NOTE: x shape is [Batch size * # proposals for each image, 512 (embedding dim)]
            x = F.normalize(x, p=2, dim=1)
            if self.use_regional_embedding:
                # NOTE: gradient of cloned tensor will be propagated to the original tensor (x): https://discuss.pytorch.org/t/how-does-clone-interact-with-backpropagation/8247/6?u=111368
                regional_embedding = x.clone()
            # NOTE: apply normalizing temperature
            x *= self.norm_temperature

        x = torch.mm(x, zs_weight)

        if self.use_bias:
            x = x + self.cls_bias
            
        if not self.use_regional_embedding:
            return x  # class logits
        else:
            return x, regional_embedding  # class logits & regional embeddings