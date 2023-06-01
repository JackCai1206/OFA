import csv
from io import BytesIO
import os
import base64
from tqdm import tqdm
import h5py
import json
from PIL import Image
import numpy as np

data_dir = '../../dataset/OFA_data/vrd'
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
	with open(os.path.join(data_dir, f'vg_train_{version}.tsv'), 'w+', newline='\n') as f_train, \
		 open(os.path.join(data_dir, f'vg_val_{version}.tsv'), 'w+', newline='\n') as f_val:
		writer_train = csv.writer(f_train, delimiter='\t', lineterminator='\n')
		writer_val = csv.writer(f_val, delimiter='\t', lineterminator='\n')

		data = enumerate(zip(
			f['img_to_first_rel'], f['img_to_last_rel'],
			f['img_to_first_box'], f['img_to_last_box'],
			f['predicates'], f['split']))
		tqdm_obj = tqdm(data, total=len(f['split']))

		train_count = 0
		val_count = 0
		skip_count = 0
		for i, (first_rel, last_rel, first_box, last_box, preds, split) in tqdm_obj:
			if toy and ((train_count > toy_count and split == 0) or (val_count > toy_count and split != 0)):
				continue
			try:
				if last_rel - first_rel <= 0:
					skip_count += 1
					continue

				image_id = img_data[i]['image_id']
				with Image.open(os.path.join(image_dir, f'{image_id}.jpg'), 'r') as img_f:
					img_rels = f['relationships'][first_rel : last_rel+1]
					if len(img_rels) == 0:
						skip_count += 1
						continue

					pred_labels = f['predicates'][first_rel : last_rel+1].squeeze().tolist()
					boxes = f['boxes_1024'][first_box : last_box+1].squeeze().tolist()
					box_labels = f['labels'][first_box : last_box+1].squeeze().tolist()
					pred_slabels = [d['idx_to_predicate'][str(j)] for j in pred_labels]
					box_slabels = [d['idx_to_label'][str(j)] for j in box_labels]
					box_range = [first_box, last_box]

					dic = {}
					lower = int(box_range[0])
					# print(self.dataset[index][:-1])
					for j, rel in enumerate(img_rels):
						pred = pred_labels[j]
						if rel[0] not in dic:
							dic[rel[0]] = {rel[1]: pred}
						else:
							dic[rel[0]][rel[1]] = pred

					# write one data point for each object in the image and the relationships it participates in
					for sub in dic:
						objs = dic[sub].keys()
						pred_labels = dic[sub].values()
						obj_boxes = [boxes[i-lower] for i in objs]
						obj_labels = [box_labels[i-lower] for i in objs]
						pred_slabels = [d['idx_to_predicate'][str(i)] for i in pred_labels]
						obj_slabels = [d['idx_to_label'][str(i)] for i in obj_labels]
						sub_label = box_labels[sub-lower]
						sub_slabel = d['idx_to_label'][str(sub_label)]
						sub_box = boxes[sub-lower]

						row = [image_id,
	     					','.join(map(str, pred_labels)),
							','.join(map(str, objs)),
							','.join([' '.join(map(str, box)) for box in obj_boxes]),
							','.join(pred_slabels),
							','.join(obj_slabels),
							sub, sub_slabel, ' '.join(map(str, sub_box))]
						# print(row)
						if split == 0:
							if toy and train_count > toy_count:
								continue
							writer_train.writerow(row)
							train_count += 1
						else:
							if toy and val_count > toy_count:
								continue
							writer_val.writerow(row)
							val_count += 1
			except FileNotFoundError:
				print('Cannot find ' + f'{image_id}.jpg')
			# break
		print('Train:', train_count, 'Val:', val_count, 'Skipped:', skip_count)
