import csv
import os
import sys
import json
import numpy as np
from tqdm import tqdm

# Parse annotations from vg8k dataset and save them into a tsv
# 1. Relation annotations at /data/hulab/zcai75/vg8k/seed3/rel_annotations_[train|val|test].json\
# 2. Detection annotations at /data/hulab/zcai75/vg8k/seed3/detections_[train|val|test].json
# 3. Label map at /data/hulab/zcai75/vg8k/seed3/[objects|predicates].json
# 4. Each tsv line stores uniq_id, pred_ids, box_ids, box_range, img_rels, boxes, pred_label, box_label, img_str

split = sys.argv[1]

rel_anno_path = '/data/hulab/zcai75/vg8k/seed3/rel_annotations_{}.json'.format(split)
det_anno_path = '/data/hulab/zcai75/vg8k/seed3/detections_{}.json'.format(split)
obj_label_path = '/data/hulab/zcai75/vg8k/seed3/objects.json'
pred_label_path = '/data/hulab/zcai75/vg8k/seed3/predicates.json'
tsv_path = 'dataset/OFA_data/vg8k/{}.tsv'.format(split)
with open(rel_anno_path, 'r') as f1, open(det_anno_path, 'r') as f2, \
        open(obj_label_path, 'r') as f3, open(pred_label_path, 'r') as f4, open(tsv_path, 'w+') as f5:
    rel_anno = json.load(f1)
    det_anno = json.load(f2)
    obj_label = json.load(f3)
    pred_label = json.load(f4)

    writer = csv.writer(f5, delimiter='\t', lineterminator='\n')

    for fn in tqdm(rel_anno):
        uniq_id = fn.split('.')[0]
        
