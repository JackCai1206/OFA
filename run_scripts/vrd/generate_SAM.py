import csv
from io import BytesIO
import os
import base64
from tqdm import tqdm
import h5py
import json
from PIL import Image
import numpy as np
import random
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

data_dir = '/data/hulab/zcai75/OFA_data/SAM'
vg_dir = '/data/hulab/zcai75/visual_genome'
image_dir = os.path.join(vg_dir, 'VG_100K')
toy = False
version = 'toy' if toy else 'full'
toy_count = 1000

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

def get_boxes_SAM(image, generator):
    masks = generator.generate(image)
    bbox_og = bbox = [mask['bbox'] for mask in masks]
    ious = [mask['predicted_iou'] for mask in masks]
    # convert from xywh to ccwh
    for i in range(len(bbox)):
        bbox[i][0] += bbox[i][2] / 2
        bbox[i][1] += bbox[i][3] / 2
    # scale to 1024
    max_image_size = max(image.shape[0], image.shape[1])
    bbox = [[int(round(b[0] / max_image_size * 1024)), int(round(b[1] / max_image_size * 1024)), int(round(b[2] / max_image_size * 1024)), int(round(b[3] / max_image_size * 1024))] for b in bbox]
    return bbox, ious

torch.cuda.set_device(0)
sam = sam_model_registry["vit_h"](checkpoint='/data/hulab/zcai75/checkpoints/SAM/sam_vit_h_4b8939.pth')
sam.eval()
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.7, box_nms_thresh=0.7)
with h5py.File(os.path.join(data_dir, 'SAM_predictions_all.h5'), 'a') as f, \
	 open(os.path.join(vg_dir, 'image_data.json')) as img_data, h5py.File(os.path.join(vg_dir, 'VG-SGG-with-attri.h5'), 'r') as ann:
	img_data = json.load(img_data)
	#dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou','point_coords', 'stability_score', 'crop_box'])
	# f.create_dataset('image_ids', dtype=int)

	avg_num_boxes = 0
	tqdm_obj = tqdm(enumerate(img_data), total=len(img_data))
	for i, dat in tqdm_obj:
		if ann['split'][i] == 0:
			continue
		image_id = dat['image_id']
		if f.get(str(image_id)) is not None:
			continue
		with Image.open(os.path.join(image_dir, f'{image_id}.jpg'), 'r') as img_f:
			img_f = img_f.convert('RGB')
			img = np.atleast_3d(np.array(img_f))
			# print(img.shape)

			boxes, ious = get_boxes_SAM(img, mask_generator)
			boxes = np.array(boxes)
			# print(len(boxes))
			avg_num_boxes += len(boxes)
			tqdm_obj.set_postfix({'num_boxes': len(boxes)})

			# draw the bboxes and save the picture
			# import cv2
			# img = np.array(img_f)
			# max_image_size = max(img.shape[0], img.shape[1])
			# scale = max_image_size / 1024
			# for box in boxes:
			# 	box = [float(b) for b in box]
			# 	box = [box[0]*scale, box[1]*scale, box[2]*scale, box[3]*scale]
			# 	# convert from center to corner
			# 	box = [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]

			# 	box = [int(round(b)) for b in box]
			# 	cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
			# cv2.imwrite(f'{image_id}.jpg', img)

			# f['predicted_iou'][int(image_id)] = ious
			# f['bbox'][int(image_id)] = np.array(boxes).T
			f.create_dataset(f'{image_id}/predicted_iou', data=ious)
			f.create_dataset(f'{image_id}/bbox', data=np.array(boxes).T)
			f.flush()
		# break
	avg_num_boxes /= len(img_data)
	print(f'Average number of boxes: {avg_num_boxes}')
