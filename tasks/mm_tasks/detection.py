# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import code
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from operator import xor
import os
from typing import Callable, List, Optional
from argparse import Namespace
import numpy as np
from pycocotools import cocoeval

import torch
from fairseq import metrics
from fairseq.tasks import register_task
from utils.eval_utils import decode_fn
from utils.bounding_box import BBFormat, BoundingBox, CoordinatesType
from utils.coco_evaluator import get_coco_metrics, _group_detections, _compute_ious, _evaluate_image, _compute_ap_recall

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.detection_dataset import DetectionDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.info("Please install tensorboardX: pip install tensorboardX")
    SummaryWriter = None

COCO_CLS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

@dataclass
class DetectionConfig(OFAConfig):
    tensorboard_logdir: Optional[str] = field(default=None)

    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


@register_task("detection", dataclass=DetectionConfig)
class DetectionTask(OFATask):
    def __init__(self, cfg: DetectionConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        # accumulate evaluations on a per-class basis
        self._evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        self.coco_cls_tokens = [self.target_dictionary.index(COCO_CLS[i]) for i in range(len(COCO_CLS))]
        self.allowed_tokens = self.coco_cls_tokens.extend(range(len(self.src_dict) - self.cfg.num_bins, len(self.src_dict)))
        self.eval_scalar_keys = ['AP', 'total positives', 'TP', 'FP']
        self.ref_strs = []
        self.hyp_strs = []
        self.tensorboard_writer = None
        self.tensorboard_dir = None
        if self.cfg.tensorboard_logdir and SummaryWriter is not None:
            self.tensorboard_dir = os.path.join(self.cfg.tensorboard_logdir, "valid_extra")

    # Override the default behavior because we have list log outputs
    @staticmethod
    def logging_outputs_can_be_summed(criterion):
        return False

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = DetectionDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size
        )

    def build_model(self, cfg):
        def prefix_allowed_tokens(batch_id, token_ids):
            return self.allowed_tokens

        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args), prefix_allowed_tokens_fn=prefix_allowed_tokens
            )
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model

    def _calculate_scores(self, dt, gt, thresh=0.5, max_dets=100, area_range=(0, np.inf)):
        # separate bbs per image X class
        _bbs = _group_detections(dt, gt)
        # pairwise ious
        _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

        scores, matched, NP = [], [], 0
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                thresh,
                max_dets,
                area_range,
            )
            acc = self._evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])
            scores.extend(ev["scores"])
            matched.extend(ev["matched"])
            NP += ev['NP']
        return np.array(scores), np.array(matched, dtype=bool), NP


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            gt, dt = [], []
            for i in range(len(hyps['boxes'])): # for each image in the batch
                hyps['boxes'][i] = hyps['boxes'][i] / (self.cfg.num_bins - 1) * self.cfg.max_image_size
                hyps['boxes'][i][:, ::2] /= sample['w_resize_ratios'][i]
                hyps['boxes'][i][:, 1::2] /= sample['h_resize_ratios'][i]
                boxes = hyps['boxes'][i].cpu()
                cats = hyps['cats'][i]
                confs = hyps['confs'][i].cpu()
                for j in range(len(boxes)):
                    dt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], confidence=confs[j], format=BBFormat.XYX2Y2))

            for i in range(len(refs['boxes'])): # for each image in the batch
                refs['boxes'][i] = refs['boxes'][i] / (self.cfg.num_bins - 1) * self.cfg.max_image_size
                refs['boxes'][i][:, ::2] /= sample['w_resize_ratios'][i]
                refs['boxes'][i][:, 1::2] /= sample['h_resize_ratios'][i]
                boxes = refs['boxes'][i].cpu()
                cats = refs['cats'][i]
                for j in range(len(boxes)):
                    gt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], format=BBFormat.XYX2Y2))

            logging_output['dt'] = dt
            logging_output['gt'] = gt

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        
        def sum_logs(key):
            import torch
            result = sum([log.get(key, 0) for log in logging_outputs])
            if torch.is_tensor(result):
                result = result.cpu()
            return result
        
        def cat_logs(key):
            import torch
            result = np.concatenate([log.get(key, []) for log in logging_outputs])
            if torch.is_tensor(result):
                result = result.cpu()
            return result
        
        def del_logs(key):
            for log in logging_outputs:
                del log[key]

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)
        
        if logging_outputs[0].get('gt', None) != None:
            scores, matched, NP = self._calculate_scores(cat_logs('dt'), cat_logs('gt'))
            res = _compute_ap_recall(scores, matched, NP)
            for key in self.eval_scalar_keys:
                metrics.log_scalar(key, res[key])
            del_logs('dt')
            del_logs('gt')

    def post_validate(self, model, stats, agg):
        # now reduce accumulations
        for class_id in self._evals:
            acc = self._evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])
        res = {}
        # run ap calculation per-class
        for class_id in self._evals:
            ev = self._evals[class_id]
            res[class_id] = {
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
            }

        # AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
        # AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
        # AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])
        for key in self.eval_scalar_keys:
            val = np.mean([res[cls][key] for cls in res if res[cls][key] is not None]) # average across classes
            metrics.log_scalar(key, val)
        
        self._evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

        if self.tensorboard_dir:
            self.log_tensorboard(self.hyp_strs, self.ref_strs, stats['num_updates'])

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = {'cats': [], 'boxes':[], 'confs': []}, {'cats': [], 'boxes':[]}
        refs['boxes'] = [sample['boxes_targets'][i]['boxes'] for i in range(sample['nsentences'])]
        refs['cats'] = [sample['boxes_targets'][i]['labels'] for i in range(sample['nsentences'])]
        for i in range(len(gen_out)):
            out_str = gen_out[i][0]["tokens"]
            confs = gen_out[i][0]['positional_scores']
            out_str = out_str[:len(out_str)-1 - (len(out_str)-1) % 5]
            confs = confs[:len(confs)-1 - (len(confs)-1) % 5]
            box_locs = torch.arange(1, len(out_str) + 1) % 5 != 0
            boxes_flat = out_str[box_locs] - len(self.src_dict) + self.cfg.num_bins
            boxes = boxes_flat.reshape(-1, 4)
            hyps['cats'].append(out_str[4::5].clone()) # select the 5th token of every groups of 5
            hyps['boxes'].append(boxes.clone())
            hyps['confs'].append(confs.reshape(-1, 5).mean(dim=1).clone())
        if self.cfg.eval_print_samples:
            ref_str = self.bpe.decode(self.target_dictionary.string(sample['target'][0].int().cpu()))
            hyp_str = self.bpe.decode(self.target_dictionary.string(gen_out[0][0]["tokens"].int().cpu()))
            # logger.info(f"example hypothesis: {hyp_str}")
            # logger.info(f"example reference: {ref_str}")
            self.ref_strs.append(ref_str)
            self.hyp_strs.append(hyp_str)

        return hyps, refs

    def log_tensorboard(self, ref_strs, hyp_strs, num_updates, is_na_model=False):
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(self.tensorboard_dir)
        tb_writer = self.tensorboard_writer

        for hyp_str in hyp_strs:
            tb_writer.add_text('Example hypothesis', hyp_str, num_updates)
        for ref_str in ref_strs:
            tb_writer.add_text('Example reference', ref_str, num_updates)