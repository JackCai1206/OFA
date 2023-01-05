# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import code
from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from pycocotools import cocoeval

import torch
from fairseq import metrics
from fairseq.tasks import register_task
from utils.bounding_box import BBFormat, BoundingBox, CoordinatesType
from utils.coco_evaluator import get_coco_metrics

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

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

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
                for j in range(len(boxes)):
                    dt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], format=BBFormat.XYX2Y2))

            for i in range(len(refs['boxes'])): # for each image in the batch
                refs['boxes'][i] = refs['boxes'][i] / (self.cfg.num_bins - 1) * self.cfg.max_image_size
                refs['boxes'][i][:, ::2] /= sample['w_resize_ratios'][i]
                refs['boxes'][i][:, 1::2] /= sample['h_resize_ratios'][i]
                boxes = refs['boxes'][i]
                cats = refs['cats'][i]
                for j in range(len(boxes)):
                    gt.append(BoundingBox(sample['id'][i], cats[j], boxes[j], format=BBFormat.XYX2Y2))

            print(gt, dt)
            print(get_coco_metrics(gt, dt))

            # scores = self._calculate_ap_score(hyps, refs)
            # scores = [self._calculate_ap_score(hyps, box_target['boxes'].float()) for box_target in sample['boxes_targets']]
            # logging_output["_score_sum"] = scores.sum().item()
            # logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = {'cats': [], 'boxes':[]}, {'cats': [], 'boxes':[]}
        refs['boxes'] = [sample['boxes_targets'][i]['boxes'] for i in range(sample['nsentences'])]
        refs['cats'] = [sample['boxes_targets'][i]['labels'] for i in range(sample['nsentences'])]
        for i in range(len(gen_out)):
            out_str = gen_out[i][0]["tokens"]
            out_str = out_str[:len(out_str)-1 - (len(out_str)-1) % 5]
            box_locs = torch.arange(1, len(out_str) + 1) % 5 != 0
            boxes_flat = out_str[box_locs] - len(self.src_dict) + self.cfg.num_bins
            boxes = boxes_flat.reshape(-1, 4)
            hyps['cats'].append(out_str[4::5]) # select the 5th token of every groups of 5
            hyps['boxes'].append(boxes.clone())
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps)
            logger.info("example reference: ", refs)

        return hyps, refs