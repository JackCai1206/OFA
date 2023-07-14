
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
from omegaconf import DictConfig
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task
from data.mm_data.sg_cls_dataset import SGCLSDataset
from data.mm_data.vrd2_dataset import VRD2Dataset
from data.mm_data.vrd_dataset import VRDDataset

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.caption_dataset import CaptionDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class VRDConfig(OFAConfig):
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    vg_json_dir: Optional[str] = field(
        default=None, metadata={"help": "path to the directory of visual genome json files"}
    )


@register_task("vrd2", dataclass=VRDConfig)
class VRD2Task(OFATask):
    def __init__(self, cfg: VRDConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.valid_count = 0

        if cfg.vg_json_dir is not None:
            with open(cfg.vg_json_dir) as f:
                self.vg_json = json.load(f)
        else:
            with open('/home/zcai75/Github/OFA_forked/dataset/visual_genome/VG-SGG-dicts-with-attri.json') as f:
                self.vg_json = json.load(f)
    
    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )

        for symbol in ['<sub>', '<obj>', '<pred>', '<rare>']:
            src_dict.add_symbol(symbol)
            tgt_dict.add_symbol(symbol)
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("vrd setup: source dictionary: {} types".format(len(src_dict)))
        logger.info("vrd setup: target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        if split == 'train' or split == 'valid':
            self.datasets[split] = VRD2Dataset(
                split,
                dataset,
                self.bpe,
                self.src_dict,
                self.tgt_dict,
                max_src_length=self.cfg.max_src_length,
                max_tgt_length=self.cfg.max_tgt_length,
                patch_image_size=self.cfg.patch_image_size,
                num_bins=self.cfg.num_bins,
                imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std
            )
        elif split == 'test':
            self.datasets[split] = SGCLSDataset(
                split,
                dataset,
                self.bpe,
                self.src_dict,
                self.tgt_dict,
                max_src_length=self.cfg.max_src_length,
                max_tgt_length=self.cfg.max_tgt_length,
                patch_image_size=self.cfg.patch_image_size,
                num_bins=self.cfg.num_bins,
                imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std
            )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        hyps, refs = self._inference(self.sequence_generator, sample, model)
        img_ids = sample['id'].tolist()
        if self.valid_count % 50 == 0:
            print(img_ids[0], hyps[0], refs[0])

        self.valid_count += 1

        return loss, sample_size, logging_output
    
    def post_validate(self, model, stats, agg):
        self.valid_count = 0

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

    def _inference(self, generator, sample, model):

        def decode(toks):
            s = self.tgt_dict.string(
                toks.int().cpu()
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        # transtab = str.maketrans({key: None for key in string.punctuation})
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.strip())
            refs.append(
                decode(
                        utils.strip_pad(sample["target"][i], self.tgt_dict.pad())
                    ).split('&&')
            )
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs
