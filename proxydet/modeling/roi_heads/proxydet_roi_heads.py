# Copyright (c) Facebook, Inc. and its affiliates.
'''
    Modifications Copyright (c) 2024-present NAVER Corp, Apache License v2.0
    original source: https://github.com/facebookresearch/Detic/blob/main/detic/modeling/roi_heads/detic_roi_heads.py
'''
import copy
import numpy as np
import json
import math
import torch
from torch import nn
from torch.autograd.function import Function
from typing import Dict, List, Optional, Tuple, Union
from torch.nn import functional as F

from fvcore.nn import giou_loss

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.box_head import build_box_head
from .proxydet_fast_rcnn import ProxydetFastRCNNOutputLayers
from ..debug import debug_second_stage

from torch.cuda.amp import autocast
from copy import deepcopy

@ROI_HEADS_REGISTRY.register()
class ProxydetCascadeROIHeads(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        mult_proposal_score: bool = False,
        with_image_labels: bool = False,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        add_feature_to_prop: bool = False,
        mask_weight: float = 1.0,
        one_class_per_proposal: bool = False,
        use_regional_embedding: bool = False,
        base_cat_mask: str = None,
        cmm_stage: list = [],
        cmm_stage_test: list = None,
        cmm_beta: float = 1.0,
        cmm_loss: str = "l1",
        cmm_loss_weight: float = 1.0,
        cmm_separated_branch: bool = False,
        cmm_base_alpha: float = 0.5,
        cmm_novel_beta: float = 0.5,
        cmm_use_inl: bool = False,
        cmm_prototype: str = "center",
        cmm_prototype_temp: float = 1.0,
        cmm_classifier_temp: float = None,
        cmm_use_sigmoid_ce: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.with_image_labels = with_image_labels
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.add_feature_to_prop = add_feature_to_prop
        self.mask_weight = mask_weight
        self.one_class_per_proposal = one_class_per_proposal
        self.use_regional_embedding = use_regional_embedding
        self.base_cat_mask = torch.tensor(np.load(base_cat_mask)).bool()
        self.cmm_stage = cmm_stage
        self.cmm_stage_test = cmm_stage_test
        self.cmm_beta = cmm_beta
        self.cmm_loss = cmm_loss
        self.cmm_loss_weight = cmm_loss_weight
        self.cmm_separated_branch = cmm_separated_branch
        self.cmm_base_alpha = cmm_base_alpha
        self.cmm_novel_beta = cmm_novel_beta
        self.cmm_use_inl = cmm_use_inl
        self.cmm_prototype = cmm_prototype
        self.cmm_prototype_temp = cmm_prototype_temp
        self.cmm_classifier_temp = cmm_classifier_temp
        self.cmm_use_sigmoid_ce = cmm_use_sigmoid_ce

        if self.cmm_separated_branch:
            self.box_head_cmm = deepcopy(self.box_head)
            self.box_predictor_cmm = deepcopy(self.box_predictor)
            if self.cmm_classifier_temp is not None:
                for k in range(self.num_cascade_stages):
                    self.box_predictor_cmm[k].cls_score.norm_temperature = self.cmm_classifier_temp
            if not self.cmm_use_sigmoid_ce:
                for k in range(self.num_cascade_stages):
                    self.box_predictor_cmm[k].use_sigmoid_ce = self.cmm_use_sigmoid_ce # using bce or ce

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'add_feature_to_prop': cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'one_class_per_proposal': cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL,
            "use_regional_embedding": cfg.MODEL.ROI_BOX_HEAD.USE_REGIONAL_EMBEDDING,
            "base_cat_mask": cfg.MODEL.ROI_HEADS.BASE_CAT_MASK,
            "cmm_stage": cfg.MODEL.ROI_HEADS.CMM.MIXUP_STAGE,
            "cmm_stage_test": cfg.MODEL.ROI_HEADS.CMM.MIXUP_STAGE_TEST,
            "cmm_beta": cfg.MODEL.ROI_HEADS.CMM.MIXUP_BETA,
            "cmm_loss": cfg.MODEL.ROI_HEADS.CMM.LOSS,
            "cmm_loss_weight": cfg.MODEL.ROI_HEADS.CMM.LOSS_WEIGHT,
            "cmm_separated_branch": cfg.MODEL.ROI_HEADS.CMM.SEPARATED_BRANCH,
            "cmm_base_alpha": cfg.MODEL.ROI_HEADS.CMM.BASE_ALPHA,
            "cmm_novel_beta": cfg.MODEL.ROI_HEADS.CMM.NOVEL_BETA,
            "cmm_use_inl": cfg.MODEL.ROI_HEADS.CMM.USE_INL,
            "cmm_prototype": cfg.MODEL.ROI_HEADS.CMM.PROTOTYPE,
            "cmm_prototype_temp": cfg.MODEL.ROI_HEADS.CMM.PROTOTYPE_TEMP,
            "cmm_classifier_temp": cfg.MODEL.ROI_HEADS.CMM.CLASSIFIER_TEMP,
            "cmm_use_sigmoid_ce": cfg.MODEL.ROI_HEADS.CMM.USE_SIGMOID_CE,
        })
        return ret


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                ProxydetFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret

    def _embed_interp(self, re_i, re_j, te_i, te_j, lam):
        # mix image, text embedding
        _mixed_re = lam * re_i + (1 - lam) * re_j
        _mixed_te = lam * te_i + (1 - lam) * te_j
        return _mixed_re, _mixed_te

    def _get_head_outputs(self, features, proposals, image_sizes, _run_stage, box_predictor, targets=None, ann_type='box', classifier_info=(None,None,None)):
        head_outputs = []  # (predictor, predictions, proposals)
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training and ann_type in ['box']:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = _run_stage(features, proposals, k, 
                classifier_info=classifier_info)
            prev_pred_boxes = box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((box_predictor[k], predictions, proposals))
        return head_outputs, proposals

    def loss_mixup(self, mixed_re, mixed_te, text_embeddings):
        # loss
        if self.cmm_loss in ["l1", "l2"]:
            if self.cmm_loss == "l1":
                loss_type = F.l1_loss
            elif self.cmm_loss == "l2":
                loss_type = F.mse_loss
            cmm_loss = loss_type(mixed_re, mixed_te)
        else:
            raise ValueError("No such loss is supported : ", self.cmm_loss)
        return cmm_loss

    def mixup(self, stage, regional_embeddings, text_embeddings, gt_classes, proto_weights=None):
        # class-wise multi-modal mixup
        try:
            neg_class = text_embeddings.shape[0] - 1

            # select positive text embeddings
            all_classes = torch.unique(gt_classes)
            pos_classes = all_classes[all_classes != neg_class]

            # select class-wise regional embeddings & text embeddings
            clswise_re = []
            clswise_te = []
            for p_c in pos_classes:
                mask = (gt_classes == p_c)
                if self.cmm_prototype == "center":
                    _clswise_re = torch.mean(regional_embeddings[mask], axis=0, keepdim=True)
                elif self.cmm_prototype in ["obj_score", "iou"]:
                    soft_proto_weights = F.softmax(proto_weights[mask] / self.cmm_prototype_temp, dim=0)
                    _clswise_re = torch.sum(regional_embeddings[mask] * soft_proto_weights.unsqueeze(-1), 0, keepdim=True)
                _clswise_te = text_embeddings[int(p_c.item())].unsqueeze(0)
                clswise_re.append(_clswise_re)
                clswise_te.append(_clswise_te)
            
            if len(clswise_re) == 0:
                raise ValueError("no positive base classes found for mixup.")

            clswise_re = torch.cat(clswise_re, dim=0)
            clswise_re = F.normalize(clswise_re, p=2, dim=1) # re-normalize
            clswise_te = torch.cat(clswise_te, dim=0)

            if self.cmm_beta == 0:
                lam = float(np.random.randint(2))
            else:
                lam = np.random.beta(self.cmm_beta, self.cmm_beta)

            # random shuffle for mixup pair
            rand_index = torch.randperm(clswise_re.size()[0]).to(clswise_re.device)

            # mixup
            sf_clswise_re = clswise_re[rand_index]
            sf_clswise_te = clswise_te[rand_index]
            mixed_re, mixed_te = self._embed_interp(clswise_re, sf_clswise_re, clswise_te, sf_clswise_te, lam)
            mixed_re = F.normalize(mixed_re, p=2, dim=1)
            mixed_te = F.normalize(mixed_te, p=2, dim=1)
            cmm_loss = self.loss_mixup(mixed_re, mixed_te, text_embeddings) 

        except Exception as e:
            print("Caught this error in mixup: " + repr(e), "Thus skipping current batch w/o mixup...")
            cmm_loss = text_embeddings[0].new_zeros([1])[0]

        return cmm_loss

    def _forward_box(self, features, proposals, targets=None, 
        ann_type='box', classifier_info=(None,None,None)):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        # head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        head_outputs, proposals = self._get_head_outputs(features, proposals, image_sizes, self._run_stage, self.box_predictor, targets, ann_type, classifier_info)
        if self.cmm_separated_branch:
            # separated forward
            head_outputs_cmm, proposals_cmm = self._get_head_outputs(features, proposals, image_sizes, self._run_stage_cmm, self.box_predictor_cmm, targets, ann_type, classifier_info)
        
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    if ann_type != 'box': 
                        stage_losses = {}
                        if ann_type in ['image', 'caption', 'captiontag']:
                            image_labels = [x._pos_category_ids for x in targets]
                            weak_losses = predictor.image_label_losses(
                                predictions, proposals, image_labels,
                                classifier_info=classifier_info,
                                ann_type=ann_type)
                            stage_losses.update(weak_losses)

                            if self.cmm_use_inl and len(self.cmm_stage) > 0 and stage in self.cmm_stage:
                                if self.cmm_separated_branch:
                                    # get regional embeddings (l2 normalized) from separated branch
                                    regional_embeddings = head_outputs_cmm[stage][1][2]
                                else:
                                    # get regional embeddings (l2 normalized)
                                    regional_embeddings = predictions[2]

                                # get text embeddings (L2 normalized), [C (1203 + 1), embedding dim]
                                text_embeddings = predictor.cls_score.zs_weight.t()

                                # get max-size proposal's regional embedding, per image
                                num_inst_per_image = [len(p) for p in proposals_cmm]
                                re_per_image = regional_embeddings.split(num_inst_per_image, dim=0)

                                maxsize_re_per_image = []
                                for p, re in zip(proposals_cmm, re_per_image):
                                    sizes = p.proposal_boxes.area()
                                    ind = sizes[:-1].argmax().item() if len(sizes) > 1 else 0
                                    maxsize_re_per_image.append(re[ind].unsqueeze(0))
                                maxsize_re_per_image = torch.cat(maxsize_re_per_image, dim=0)
                                maxsize_re_per_image = maxsize_re_per_image.to(regional_embeddings.device)

                                # get gt classes per max-size proposal
                                # TODO: add best-label per image by cls loss from weak_losses (image-label loss)
                                # NOTE: image_labels are not multi-labels.
                                gt_classes = (
                                    regional_embeddings.new_tensor(
                                        [np.random.choice(labels, 1, replace=False)[0] for labels in image_labels],
                                        dtype=torch.long
                                    ) if len(proposals_cmm)
                                    else torch.empty(0)
                                )

                                proto_weights = None

                                # get text embeddings (L2 normalized), [C (1203 + 1), embedding dim]
                                text_embeddings = predictor.cls_score.zs_weight.t()
                                cmm_loss = self.mixup(stage, maxsize_re_per_image, text_embeddings, gt_classes, proto_weights)
                                stage_losses["cmm_image_loss"] = (
                                    cmm_loss * self.cmm_loss_weight
                                )
                                stage_losses["cmm_loss"] = \
                                    predictions[0].new_zeros([1])[0]

                    else: # supervised
                        stage_losses = predictor.losses(
                            (predictions[0], predictions[1]), proposals,
                            classifier_info=classifier_info)
                        if self.with_image_labels:
                            stage_losses['image_loss'] = \
                                predictions[0].new_zeros([1])[0]

                        if len(self.cmm_stage) > 0 and stage in self.cmm_stage:
                            assert self.use_regional_embedding

                            # get gt classes per proposal
                            # e.g. dtype: torch.int64, value: tensor([ 142,  111,  142,  ..., 1203, 1203, 1203], device='cuda:6')
                            gt_classes = (
                                cat([p.gt_classes for p in proposals_cmm], dim=0)
                                if len(proposals_cmm)
                                else torch.empty(0)
                            )

                            if self.cmm_prototype in ["obj_score"]:
                                proto_weights = (
                                    cat([p.objectness_logits for p in proposals_cmm], dim=0)
                                    if len(proposals_cmm)
                                    else torch.empty(0)
                                )
                            elif self.cmm_prototype in ["iou"]:
                                gt_boxes = (
                                    cat([p.gt_boxes.tensor for p in proposals_cmm], dim=0)
                                    if len(proposals_cmm)
                                    else torch.empty(0)
                                )
                                proposal_boxes = (
                                    cat([p.proposal_boxes.tensor for p in proposals_cmm], dim=0)
                                    if len(proposals_cmm)
                                    else torch.empty(0)
                                )

                                proto_weights = 1 - giou_loss(proposal_boxes, gt_boxes, reduction="none") # GIoU. (-1 < x < 1)
                            else:
                                proto_weights = None

                            if self.cmm_separated_branch:
                                # get regional embeddings (l2 normalized) from separated branch
                                regional_embeddings = head_outputs_cmm[stage][1][2]
                            else:
                                # get regional embeddings (l2 normalized)
                                regional_embeddings = predictions[2]

                            # get text embeddings (L2 normalized), [C (1203 + 1), embedding dim]
                            text_embeddings = predictor.cls_score.zs_weight.t()

                            cmm_loss = self.mixup(stage, regional_embeddings, text_embeddings, gt_classes, proto_weights)
                            stage_losses["cmm_loss"] = (
                                cmm_loss * self.cmm_loss_weight
                            )

                            if self.cmm_use_inl:
                                stage_losses["cmm_image_loss"] = \
                                        predictions[0].new_zeros([1])[0]
                            
                losses.update({k + "_stage{}".format(stage): v \
                    for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]

            if self.cmm_separated_branch:
                # scores from separated branch
                if self.cmm_stage_test is None:
                    # average all stage's classification scores
                    scores_per_stage_cmm = [h[0].predict_probs(h[1], h[2]) for h in head_outputs_cmm]
                    scores_cmm = [
                        sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                        for scores_per_image in zip(*scores_per_stage_cmm)
                    ]
                else:
                    # only using specific stages
                    scores_per_stage_cmm = [h[0].predict_probs(h[1], h[2]) for k, h in enumerate(head_outputs_cmm) if k in self.cmm_stage_test]
                    scores_cmm = [
                        sum(list(scores_per_image)) * (1.0 / len(self.cmm_stage_test))
                        for scores_per_image in zip(*scores_per_stage_cmm)
                    ]

                base_cat_mask = self.base_cat_mask
                assert len(scores) == 1
                bg_score = scores[0][:, -1].clone()
                scores[0][:, base_cat_mask] = scores[0][:, base_cat_mask].pow(
                    1.0 - self.cmm_base_alpha
                ) * scores_cmm[0][:, base_cat_mask].pow(self.cmm_base_alpha)
                scores[0][:, ~base_cat_mask] = scores[0][:, ~base_cat_mask].pow(
                    1.0 - self.cmm_novel_beta
                ) * scores_cmm[0][:, ~base_cat_mask].pow(self.cmm_novel_beta)
                scores[0][:, -1] = bg_score

            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]
            if self.one_class_per_proposal:
                scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(
                (predictions[0], predictions[1]), proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances


    def forward(self, images, features, proposals, targets=None,
        ann_type='box', classifier_info=(None,None,None)):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if self.training:
            if ann_type in ['box', 'prop', 'proptag']:
                proposals = self.label_and_sample_proposals(
                    proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)
            
            losses = self._forward_box(features, proposals, targets, \
                ann_type=ann_type, classifier_info=classifier_info)
            if ann_type == 'box' and targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals)
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
                losses.update(self._forward_keypoint(features, proposals))
            else:
                losses.update(self._get_empty_mask_loss(
                    features, proposals,
                    device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, classifier_info=classifier_info)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals


    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2., 
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.), 
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])


    def _get_empty_mask_loss(self, features, proposals, device):
        if self.mask_on:
            return {'loss_mask': torch.zeros(
                (1, ), device=device, dtype=torch.float32)[0]}
        else:
            return {}


    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        """
        Add objectness_logits
        """
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(
            boxes, image_sizes, logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals


    def _run_stage(self, features, proposals, stage, \
        classifier_info=(None,None,None)):
        """
        Support classifier_info and add_feature_to_prop
        """
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        if self.add_feature_to_prop:
            feats_per_image = box_features.split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat
        return self.box_predictor[stage](
            box_features, 
            classifier_info=classifier_info)

    def _run_stage_cmm(self, features, proposals, stage, \
        classifier_info=(None,None,None)):
        """
        Support classifier_info and add_feature_to_prop
        """
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head_cmm[stage](box_features)
        if self.add_feature_to_prop:
            feats_per_image = box_features.split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat
        return self.box_predictor_cmm[stage](
            box_features, 
            classifier_info=classifier_info)
