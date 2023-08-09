from datasets import load_dataset
import os
import csv
import h5py
import random
from tqdm import tqdm
import json

ds_rel = load_dataset('visual_genome', 'relationships_v1.2.0', cache_dir='/data/hulab/zcai75/visual_genome')
ds_attr = load_dataset('visual_genome', 'attributes_v1.2.0', cache_dir='/data/hulab/zcai75/visual_genome')


data_dir = '../../dataset/OFA_data/vg_full'
vg_dir = '/data/hulab/zcai75/visual_genome'
image_dir = os.path.join(vg_dir, 'VG_100K')

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

def get_box(box):
    return ' '.join(map(str, [box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']]))

is_train = {}
with h5py.File(os.path.join(vg_dir, 'VG-SGG-with-attri.h5'), 'r') as f, \
    open(os.path.join(vg_dir, 'image_data.json')) as img_data:
    img_data = json.load(img_data)
    is_train = {img_data[i]['image_id']: f['split'][i] == 0 for i in range(len(img_data))}

train_count = 0
val_count = 0
with open(os.path.join(data_dir, 'train.tsv'), 'w+') as train_file, \
     open(os.path.join(data_dir, 'val.tsv'), 'w+') as val_file:
    train_writer = csv.writer(train_file, delimiter='\t')
    val_writer = csv.writer(val_file, delimiter='\t')
    ds_count = len(ds_rel['train'])
    indices = list(range(ds_count))
    random.shuffle(indices)
    for i in tqdm(indices):
        sample = ds_rel['train'][i]
        img_id = sample['image_id']
        for rel in sample['relationships']:
            row = [str(img_id) + '-' + str(rel['relationship_id']),
                   get_box(rel['object']), get_box(rel['subject']),
                   rel['object']['names'][0], rel['subject']['names'][0], 
                rel['predicate'].lower()]
            if img_id not in is_train or is_train[img_id]:
                train_writer.writerow(row)
                train_count += 1
            else:
                val_writer.writerow(row)
                val_count += 1
            # print(rel)
        # break

print(train_count, val_count)

# with open(os.path.join(data_dir, 'train.tsv'), 'r') as train_file, \
#      open(os.path.join(data_dir, 'val.tsv'), 'r') as val_file:
