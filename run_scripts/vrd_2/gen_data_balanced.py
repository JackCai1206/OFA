import csv
from io import BytesIO
import math
import os
import base64
import shutil
from tqdm import tqdm
import h5py
import json
from PIL import Image
import numpy as np

og_data_dir = '../../dataset/OFA_data/vrd_2'
data_dir = '../../dataset/OFA_data/vrd2_balanced'
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
	d['predicate_count']
	# calculate oversampling ratio from predicate count for each predicate and store in dict
	pred_count = d['predicate_count']
	pred_ratio = {}
	for pred in pred_count:
		pred_ratio[pred] = sum(pred_count.values()) / pred_count[pred]
	# search for the right coefficient such that each ratios is greater than 1
	# this is the probability of sampling a predicate
	pred_ratio = {k: v / min(pred_ratio.values()) for k, v in pred_ratio.items()}
	print(pred_ratio)

	# shutil.copyfile(os.path.join(og_data_dir, f'vg_train_{version}.tsv'), os.path.join(data_dir, f'vg_train_{version}.tsv'))
	# shutil.copyfile(os.path.join(og_data_dir, f'vg_val_{version}.tsv'), os.path.join(data_dir, f'vg_val_{version}.tsv'))

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	with open(os.path.join(og_data_dir, f'vg_train_{version}.tsv'), 'r', newline='\n') as f_train_og, \
		 open(os.path.join(data_dir, f'vg_train_{version}_extra.tsv'), 'w+', newline='\n') as f_train_extra:
		reader = csv.reader(f_train_og, delimiter='\t')
		writer = csv.writer(f_train_extra, delimiter='\t')

		# get the number of rows in the file
		num_rows = sum(1 for row in reader)
		print(f'num_rows: {num_rows}')
		f_train_og.seek(0)
		count = 0
		for row in tqdm(reader):
			image_id, sub_box, obj_box, sub_slabel, obj_slabel, pred_slabel = row

			# calculate the oversamplig ratio for the sample
			os_ratio = pred_ratio[pred_slabel] / 10

			# print(os_ratio)
			for i in range(min(math.ceil(os_ratio), 10)):
				writer.writerow(row)
				count += 1

		print('Total:', count)

	# Combine the original train file and the extra rows and shuffle the rows
	count = 0
	with open(os.path.join(data_dir, f'vg_train_{version}.tsv'), 'w+', newline='\n') as f_train_shuffled, \
		open(os.path.join(data_dir, f'vg_train_{version}_extra.tsv'), 'r', newline='\n') as f_train_extra:
			reader_extra = csv.reader(f_train_extra, delimiter='\t')
			writer = csv.writer(f_train_shuffled, delimiter='\t')
			rows = [row for row in reader_extra]
			np.random.shuffle(rows)
			for row in tqdm(rows):
				writer.writerow(row)
				count += 1
	print('Total rows: ', count)

	# delete the extra file
	os.remove(os.path.join(data_dir, f'vg_train_{version}_extra.tsv'))
