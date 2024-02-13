"""
Copyright (c) 2024-present NAVER Corp.
Apache License v2.0
"""
import torch
import json
import copy
import sys
from lvis import LVIS
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from lvis import LVISEval, LVISResults

metadata = MetadataCatalog.get("lvis_v1_val")
lvis_json_file = "../datasets/lvis/lvis_v1_val.json"
lvis_gt = LVIS(lvis_json_file)

inference_result_path = sys.argv[1]
print(f"evaluating {inference_result_path}")

iou_type = "segm"

with open(inference_result_path) as fin:
    lvis_results = json.load(fin)

if iou_type == "segm":
    lvis_results = copy.deepcopy(lvis_results)
    # When evaluating mask AP, if the results contain bbox, LVIS API will
    # use the box area as the area of the instance, instead of the mask area.
    # This leads to a different definition of small/medium/large.
    # We remove the bbox field to let mask AP use mask area.
    for c in lvis_results:
        c.pop("bbox", None)

lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=300)
lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type=iou_type)
print("evaluating...")
lvis_eval.evaluate()
print("accumulatig...")
lvis_eval.accumulate()
print("summarizing...")
lvis_eval.summarize()
lvis_eval.print_results()

from nltk.corpus import wordnet

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")
nltk.download("omw-1.4")

with open(lvis_json_file) as fin:
    data = json.load(fin)
synset2cat = {x["synset"]: x for x in data["categories"]}
with open("../datasets/metadata/imagenet_lvis_wnid.txt") as fin:
    imagenet_lvis_synsets = fin.readlines()
    imagenet_lvis_synsets = [l.rstrip("\n") for l in imagenet_lvis_synsets]

imagenet_lvis_cat_ids = []
for in_synset in imagenet_lvis_synsets:
    imagenet_lvis_cat_ids.append(synset2cat[wordnet.synset_from_pos_and_offset("n", int(in_synset[1:])).name()]['id'] - 1)
imagenet_not_lvis_cat_ids = [i for i in range(len(metadata.thing_classes)) if i not in imagenet_lvis_cat_ids]

import numpy as np
def print_summary(summary_type, freq_group_idx, imagenet_ids, iou_thr=None, area_rng="all"):
    aidx = [
        idx
        for idx, _area_rng in enumerate(lvis_eval.params.area_rng_lbl)
        if _area_rng == area_rng
    ]

    if summary_type == 'ap':
        # (iou_thres, recall, categories, freq_grp, area_rng)
        s = lvis_eval.eval["precision"]
        if iou_thr is not None:
            tidx = np.where(iou_thr == lvis_eval.params.iou_thrs)[0]
            s = s[tidx]

        cat_ids = [i for i in range(len(metadata.thing_classes))]
        if freq_group_idx is not None:
            cat_ids = lvis_eval.freq_groups[freq_group_idx]

        if imagenet_ids is not None:
            cat_ids = list(filter(lambda x: x in imagenet_ids, cat_ids))

        s = s[:, :, cat_ids, aidx]
    else:
        s = lvis_eval.eval["recall"]
        if iou_thr is not None:
            tidx = np.where(iou_thr == lvis_eval.params.iou_thrs)[0]
            s = s[tidx]
        s = s[:, :, aidx]

    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])

    print(f"{summary_type}: {mean_s*100:0.3f}")

# Adding base category which is common + frequent
if len(lvis_eval.freq_groups) < 4:
    lvis_eval.freq_groups.append(lvis_eval.freq_groups[1] + lvis_eval.freq_groups[2])

# {all, small, medium, large}
area_rng = "all"

# None, 0.5:0.9
# lvis_eval.params.iou_thrs
iou_thr = None

# ap, recall
summary_type = "ap"

# None, {0:rare (337), 1:common (461) 2:freqeunt (405), 3: base (461+405)}
# freq_groups = {"rare":0, "common":1, "freqeunt":2, "base":3, "all":None}
freq_groups = {"base":3, "rare":0, "all":None}

# None, imagenet_lvis_cat_ids, imagenet_not_lvis_cat_ids
cat_groups = {"in_im": imagenet_lvis_cat_ids, "not_in_im":imagenet_not_lvis_cat_ids, "all":None}


for f_grp, f_grp_i in freq_groups.items():
    for c_grp, c_grp_ids in cat_groups.items():
        print(f"frequency_group: {f_grp}, category_group: {c_grp}")
        print_summary(summary_type, f_grp_i, c_grp_ids, iou_thr, area_rng)