# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
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


class SGCLSDataset(OFADataset):
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
        uniq_id, pred_ids, box_ids, box_range, img_rels, boxes, pred_label, box_label, img_str = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        dic = {}
        lower = int(box_range.split(',')[0])
        box_label = box_label.split(',')
        pred_label = pred_label.split(',')
        # print(self.dataset[index][:-1])
        for i, rel in enumerate(img_rels.split(',')):
            rel = rel.split()
            b1 = self.bpe.encode(' {}'.format(box_label[int(rel[0]) - lower]))
            b2 = self.bpe.encode(' {}'.format(box_label[int(rel[1]) - lower]))
            pred = self.bpe.encode(' {}'.format(pred_label[i]))
            if b1 not in dic:
                dic[b1] = {b2: pred}
            else:
                dic[b1][b2] = pred
        
        caption = ""
        for b1 in dic:
            caption += "<sub> {} ".format(b1)
            for b2 in dic[b1]:
                caption += "<pred> {} <obj> {} ".format(dic[b1][b2], b2)

        # print(caption)

        caption_token_list = caption.strip().split()
        tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])

        # w_resize_ratio = self.patch_image_size / image.width
        # h_resize_ratio = self.patch_image_size / image.height
        boxes = [coord2bin(list(map(int, box.split())), 1024, self.num_bins) for box in boxes.split(',')]
        boxes_str = ' , '.join(boxes)
        
        src_item = self.encode_text(boxes_str, use_bpe=False)
        tgt_item = self.encode_text(tgt_caption, use_bpe=False)
        # print(tgt_item)
        # def decode(toks):
        #     s = self.tgt_dict.string(
        #         toks.int().cpu()
        #     )
        #     if self.bpe:
        #         s = self.bpe.decode(s)
        #     return s
        # print(decode(tgt_item))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])
        # print(len(src_item), len(target_item), len(prev_output_item))

        example = {
            "id": uniq_id,
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
