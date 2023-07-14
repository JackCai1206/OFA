from datasets import load_dataset
import os

ds_rel = load_dataset('visual_genome', 'relationships_v1.2.0', cache_dir='/data/hulab/zcai75/visual_genome')
ds_attr = load_dataset('visual_genome', 'attributes_v1.2.0', cache_dir='/data/hulab/zcai75/visual_genome')

data_dir = '../../dataset/OFA_data/vg_full'
vg_dir = '/data/hulab/zcai75/visual_genome'
image_dir = os.path.join(vg_dir, 'VG_100K')

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

with open(os.path.join(data_dir, 'vg_full_train')) as train_file, \
     open(os.path.join(data_dir, 'vg_full_val')) as val_file:
    for sample in ds_rel['train']:
        for rel in sample['relationships']:
            print(rel)
        break
