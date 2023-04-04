import base64
import os
import h5py
import json
from tqdm import tqdm
from csv import writer

tag = ''
IMAGE_DATA_FILE = '/data/hulab/zcai75/visual_genome/image_data.json'
VG_ANN_FILE = '/data/hulab/zcai75/visual_genome/VG-SGG-with-attri.h5'
VG_ANN_LABEL_FILE = '/data/hulab/zcai75/visual_genome/vg_motif_anno/VG-SGG-dicts-with-attri.json'
VG_IMAGE_DIR = '/data/hulab/zcai75/visual_genome/VG_100K'
OUTPUT_DIR = f'/data/hulab/zcai75/OFA_data/vgqa/'

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

ann = h5py.File(VG_ANN_FILE, 'r')
with open(IMAGE_DATA_FILE, 'r') as img_data_f, open(VG_ANN_LABEL_FILE, 'r') as label_f:
    img_data = json.load(img_data_f)
    label_data = json.load(label_f)
    print(ann.keys())
    print(img_data[0].keys())
    print(label_data.keys())

wtr = []
with open(f"{OUTPUT_DIR}/train{tag}.tsv", 'w') as f:
    wtr['train'] = writer(f, delimiter='\t')
with open(f"{OUTPUT_DIR}/val{tag}.tsv", 'w') as f:
    wtr['val'] = writer(f, delimiter='\t')

for img in tqdm(img_data):
    img_id = img['image_id']
    with open(os.path.join(VG_IMAGE_DIR, str()) + '.jpg', 'rb') as fid:
        img_str = base64.urlsafe_b64encode(fid.read()).decode('utf-8')
    split = 'train' if ann[img_id]['split'] == 1 else 'val'

    first_box = ann['img_to_first_box'][img_id]
    last_box = ann['img_to_last_box'][img_id]
    img_boxes = ann['boxes_1024'][first_box : last_box+1]
    box_labels = [label_data['idx_to_label'][str(ann['labels'][i][0])] for i in range(first_box, last_box+1)]
    box_list = '&&'.join([
        ','.join(
            ','.join(img_boxes[i]),
            list(range(first_box, last_box + 1))[i],
            box_labels[i]
        )
    for i in range(len(img_boxes))])

    first_rel = ann['img_to_first_rel'][img_id]
    last_rel = ann['img_to_last_rel'][img_id]
    img_rels = ann['relationships'][first_rel : last_rel+1]
    if len(img_rels) == 0:
        skipped = True
        continue
    rel_list = '&&'.join(
        ','.join(img_rels[i])
    for i in range(len(img_rels)))

    pred_ids = ann['predicates'][first_rel : last_rel+1]
    pred_label = [label_data['idx_to_predicate'][str(i[0])] for i in pred_ids]
    pred_list = '&&'.join(pred_label)

    wtr.writerow([img_id, img_str, box_list, rel_list, pred_list])
