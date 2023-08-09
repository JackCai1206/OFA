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
import h5py

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

def coord2bin(coord_list, box_size, w, h, max_img_size, num_bins):
	# coord / box_size(1024) * max_img_size / w_or_h
	bin_list = []
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[0] / box_size * max_img_size / w * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[1] / box_size * max_img_size / h * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[2] / box_size * max_img_size / w * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[3] / box_size * max_img_size / h * (num_bins - 1)))))]
	assert '<bin_-1>' not in bin_list, 'coord2bin error!'
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
	
	all_boxes = None
	if samples[0].get("all_boxes", None) is not None:
		all_boxes = [s['all_boxes'] for s in samples]
	
	all_boxes_raw = None
	if samples[0].get("all_boxes_raw", None) is not None:
		all_boxes_raw = [s['all_boxes_raw'] for s in samples]

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
		"all_boxes": all_boxes,
		"all_boxes_raw": all_boxes_raw
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

		self.sam_box_ds = h5py.File('/data/hulab/zcai75/OFA_data/SAM/SAM_predictions_all.h5', 'r')

		if type(bpe).__name__ == 'GPT2BPE':
			self.prompt = " What are the relations in the image?"
		elif type(bpe).__name__ == 'BertBPE':
			self.prompt = "图片描述了什么内容?"

	def __getitem__(self, index):
		uniq_id, pred_ids, box_ids, box_range, img_rels, boxes, pred_label, box_label, img_str = self.dataset[index]

		image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
		patch_image = self.patch_resize_transform(image)
		patch_mask = torch.tensor([True])

		img_rels = np.array([rels.split() for rels in img_rels.split(',')])
		lower = int(box_range.split(',')[0])
		max_img_size = max(image.width, image.height)

		if True:
			rois = [list(map(int, box.split())) for box in boxes.split(',')]
			box_ids = sorted(set(img_rels[:, 0].tolist() + img_rels[:, 1].tolist()))
			boxes_raw = boxes = {i: rois[int(i)-lower] for i in box_ids}
			boxes = {i: [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2] for i, box in boxes.items()}
			boxes = {i: coord2bin(box, 1024, image.width, image.height, max_img_size, self.num_bins) for i, box in boxes.items()}
			boxes_str = 'Describe the relations between these objects: ' + ', '.join(boxes.values())
			# print(boxes_str)	
			all_boxes = list(boxes.values())
			all_boxes_raw = list(boxes_raw.values())
		else:
			rois_sam = np.array(self.sam_box_ds[f'{uniq_id}/bbox']).T

			thresh = 0.993
			box_ids_sam = np.where(np.array(self.sam_box_ds[f'{uniq_id}/predicted_iou']) > thresh)[0]
			while box_ids_sam.shape[0] < 2:
				box_ids_sam = np.where(np.array(self.sam_box_ds[f'{uniq_id}/predicted_iou']) > thresh)[0]
				thresh -= 0.001
			while box_ids_sam.shape[0] > 10:
				box_ids_sam = np.where(np.array(self.sam_box_ds[f'{uniq_id}/predicted_iou']) > thresh)[0]
				thresh += 0.001
			boxes_raw_sam = {i: rois_sam[int(i)] for i in box_ids_sam}
			boxes_sam = {i: [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2] for i, box in boxes_raw_sam.items()}
			boxes_sam = {i: coord2bin(box, 1024, image.width, image.height, max_img_size, self.num_bins) for i, box in boxes_sam.items()}

			# all_boxes = [[box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2] for box in rois]
			# all_boxes = [coord2bin(box, 1024, image.width, image.height, max_img_size, self.num_bins) for box in all_boxes]
			# all_boxes = [self.encode_text(b, use_bpe=False) for b in all_boxes]
			# all_boxes = [self.encode_text(b, use_bpe=False) for b in boxes.values()]
			all_boxes = list(boxes_sam.values())
			all_boxes_raw = list(boxes_raw_sam.values())

		dic = {}
		box_label = box_label.split(',')
		pred_label = pred_label.split(',')
		# print(self.dataset[index][:-1])
		for i, rel in enumerate(img_rels):
			pred = self.bpe.encode(' {}'.format(pred_label[i]))
			if rel[0] not in dic:
				dic[rel[0]] = {rel[1]: pred}
			else:
				dic[rel[0]][rel[1]] = pred
		
		caption = ""
		for r1 in dic:
			l1 = self.bpe.encode(' {}'.format(box_label[int(r1) - lower]))
			caption += "<sub> {} {} ".format(l1, boxes[r1])
			for r2 in dic[r1]:
				l2 = self.bpe.encode(' {}'.format(box_label[int(r2) - lower]))
				caption += "<pred> {} <obj> {} {} ".format(dic[r1][r2], l2, boxes[r2])

		# print(caption)

		caption_token_list = caption.strip().split()
		tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
		
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
		# src_item = torch.cat([self.bos_item, self.eos_item])
		target_item = torch.cat([tgt_item, self.eos_item])
		prev_output_item = torch.cat([self.bos_item, tgt_item])
		# print(len(src_item), len(target_item), len(prev_output_item))

		example = {
			"id": uniq_id,
			"source": src_item,
			"patch_image": patch_image,
			"patch_mask": patch_mask,
			"target": target_item,
			"prev_output_tokens": prev_output_item,
			"all_boxes": all_boxes,
			"all_boxes_raw": all_boxes_raw,
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
