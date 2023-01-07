# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import code
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
import numpy as np
from pycocotools import cocoeval

import torch
from fairseq import metrics
from fairseq.tasks import register_task
from utils.bounding_box import BBFormat, BoundingBox, CoordinatesType
from utils.coco_evaluator import get_coco_metrics, _group_detections, _compute_ious, _evaluate_image, _compute_ap_recall

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.detection_dataset import DetectionDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig(OFAConfig):
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
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
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
        return scores, matched, NP


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
                boxes = hyps['boxes'][i]
                cats = hyps['cats'][i]
                confs = hyps['confs'][i]
                for j in range(len(boxes)):
                    dt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], confidence=confs[j], format=BBFormat.XYX2Y2))

            for i in range(len(refs['boxes'])): # for each image in the batch
                refs['boxes'][i] = refs['boxes'][i] / (self.cfg.num_bins - 1) * self.cfg.max_image_size
                refs['boxes'][i][:, ::2] /= sample['w_resize_ratios'][i]
                refs['boxes'][i][:, 1::2] /= sample['h_resize_ratios'][i]
                boxes = refs['boxes'][i]
                cats = refs['cats'][i]
                for j in range(len(boxes)):
                    gt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], format=BBFormat.XYX2Y2))

            scores, matched, NP = self._calculate_scores(dt, gt)
            logging_output['scores'] = scores
            logging_output['matched'] = matched
            logging_output['NP'] = NP

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
            # len(logging_outputs[0].get('scores'))
            result = np.concatenate([log.get(key, 0) for log in logging_outputs])
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)
        
        if logging_outputs[0].get('NP', None) != None:
            res = _compute_ap_recall(cat_logs("scores"), cat_logs("matched"), sum_logs("NP"))
            for key in res:
                if np.isscalar(res[key]):
                    metrics.log_scalar(key, res[key])

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
        print(res)

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
            logger.info("example hypothesis: ", hyps)
            logger.info("example reference: ", refs)

        return hyps, refs