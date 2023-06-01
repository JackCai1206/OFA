# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import os
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def coord2bin(coord_list, box_size, num_bins):
	bin_list = []
	bin_list += ["<bin_{}>".format(int(round(coord_list[0] / box_size * (num_bins - 1))))]
	bin_list += ["<bin_{}>".format(int(round(coord_list[1] / box_size * (num_bins - 1))))]
	bin_list += ["<bin_{}>".format(int(round(coord_list[2] / box_size * (num_bins - 1))))]
	bin_list += ["<bin_{}>".format(int(round(coord_list[3] / box_size * (num_bins - 1))))]
	return ' '.join(bin_list)

def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


class VRDDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        num_bins=480,
        imagenet_default_mean_and_std=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " What are the relations in the image?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"

    def __getitem__(self, index):
        image_id, pred_labels, objs, obj_boxes, pred_slabels, obj_slabels, sub, sub_slabel, sub_box = self.dataset[index]
        # print(self.dataset[index])
        pred_labels = pred_labels.split(',')
        objs = objs.split(',')
        obj_boxes = [box.split() for box in obj_boxes.split(',')]
        pred_slabels = pred_slabels.split(',')
        obj_slabels = obj_slabels.split(',')
        sub_box = sub_box.split()

        # TODO: harcoded for now
        image_dir = '/data/hulab/zcai75/visual_genome/VG_100K'
        image = Image.open(os.path.join(image_dir, f'{image_id}.jpg'), 'r')
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        sub_box = [coord2bin(list(map(int, sub_box)), 1024, self.num_bins)]
        sub_box_str = ' '.join(sub_box)
        src_item = self.encode_text(sub_box_str, use_bpe=False)
        # print(len(src_item))
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        sub_slabel = self.bpe.encode(f' {sub_slabel}')
        obj_slabels = [self.bpe.encode(f' {obj_slabel}') for obj_slabel in obj_slabels]
        pred_slabels = [self.bpe.encode(f' {pred_slabel}') for pred_slabel in pred_slabels]
        caption = "<sub> {} ".format(sub_slabel)
        for obj_slabel, pred_slabel, pred_label in zip(obj_slabels, pred_slabels, pred_labels):
            if pred_label > 9:
                caption += "<pred> <rare> {} <obj> {} ".format(pred_slabel, obj_slabel)
                print("rare predicate: {} in image {}".format(pred_label, image_id))
            else:
                caption += "<pred> {} <obj> {} ".format(pred_slabel, obj_slabel)

        caption_token_list = caption.strip().split()
        tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        tgt_item = self.encode_text(tgt_caption, use_bpe=False)
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id + '-' + str(sub),
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
