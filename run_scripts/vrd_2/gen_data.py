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

data_dir = '../../dataset/OFA_data/vrd_2'
vg_dir = '/data/hulab/zcai75/visual_genome'
image_dir = os.path.join(vg_dir, 'VG_100K')
toy = False
version = 'toy' if toy else 'full'
toy_count = 1000

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

with h5py.File(os.path.join(vg_dir, 'VG-SGG-with-attri.h5'), 'r') as f, \
	 open(os.path.join(vg_dir, 'VG-SGG-dicts-with-attri.json'), 'r') as d, \
	 open(os.path.join(vg_dir, 'image_data.json')) as img_data:
	d = json.load(d)
	img_data = json.load(img_data)
	print(f.keys())
	# print(f['boxes_1024'][0])


	data = enumerate(zip(
		f['img_to_first_rel'], f['img_to_last_rel'],
		f['img_to_first_box'], f['img_to_last_box'],
		f['predicates'], f['split']))
	tqdm_obj = tqdm(data, total=len(f['split']))

	train_count = 0
	val_count = 0
	skip_count = 0
	train_rows = []
	val_rows = []
	for i, (first_rel, last_rel, first_box, last_box, preds, split) in tqdm_obj:
		if toy and ((train_count > toy_count and split == 0) or (val_count > toy_count and split != 0)):
			continue
		try:
			if last_rel - first_rel < 0:
				skip_count += 1
				continue

			image_id = img_data[i]['image_id']
			with Image.open(os.path.join(image_dir, f'{image_id}.jpg'), 'r') as img_f:
				img_rels = f['relationships'][first_rel : last_rel+1]

				pred_labels = np.atleast_1d(f['predicates'][first_rel : last_rel+1].squeeze()).tolist()
				boxes = f['boxes_1024'][first_box : last_box+1].squeeze().tolist()
				box_labels = np.atleast_1d(f['labels'][first_box : last_box+1].squeeze()).tolist()
				pred_slabels = [d['idx_to_predicate'][str(j)] for j in pred_labels]
				box_slabels = [d['idx_to_label'][str(j)] for j in box_labels]

				for rel_i, rel in enumerate(img_rels):
					i1 = rel[0] - first_box
					i2 = rel[1] - first_box
					row = [str(image_id) + '-' + str(rel_i), ' '.join(map(str, boxes[i1])), ' '.join(map(str, boxes[i2])), box_slabels[i1], box_slabels[i2], pred_slabels[rel_i]]

					# print(row)
					if split == 0:
						if toy and train_count > toy_count:
							continue
						train_rows.append(row)
						train_count += 1
					else:
						if toy and val_count > toy_count:
							continue
						val_rows.append(row)
						val_count += 1
		except FileNotFoundError:
			print('Cannot find ' + f'{image_id}.jpg')
		# break

	with open(os.path.join(data_dir, f'vg_train_{version}.tsv'), 'w+', newline='\n') as f_train, \
		open(os.path.join(data_dir, f'vg_val_{version}.tsv'), 'w+', newline='\n') as f_val:
		writer_train = csv.writer(f_train, delimiter='\t', lineterminator='\n')
		writer_val = csv.writer(f_val, delimiter='\t', lineterminator='\n')

		random.shuffle(train_rows)
		random.shuffle(val_rows)
		for row in train_rows:
			writer_train.writerow(row)
		for row in val_rows:
			writer_val.writerow(row)
	print('Train:', train_count, 'Val:', val_count, 'Skipped:', skip_count)
