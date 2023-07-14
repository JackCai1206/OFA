import csv
import os
import sys
import json
from typing import DefaultDict
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
tsv_path = '../../dataset/OFA_data/vg8k/{}.tsv'.format(split)
if not os.path.exists(os.path.dirname(tsv_path)):
    os.makedirs(os.path.dirname(tsv_path))
with open(rel_anno_path, 'r') as f1, open(det_anno_path, 'r') as f2, \
        open(obj_label_path, 'r') as f3, open(pred_label_path, 'r') as f4, open(tsv_path, 'w') as f5:
    rel_anno = json.load(f1)
    det_anno = json.load(f2)
    obj_label = json.load(f3)
    pred_label = json.load(f4)

    writer = csv.writer(f5, delimiter='\t', lineterminator='\n')

    tqdm_obj = tqdm(rel_anno)
    line_count = 0
    tqdm_obj.set_postfix({'avg_line_count': line_count})
    for i, fn in enumerate(tqdm_obj):
        uniq_id = fn.split('.')[0]
        rels = rel_anno[fn]
        pred_labels_by_sub = DefaultDict(list)
        boxes_by_sub = DefaultDict(list)
        box_labels_by_sub = DefaultDict(list)
        pred_slabels_by_sub = DefaultDict(list)
        box_slabels_by_sub = DefaultDict(list)
        sub_labels = []
        sub_slabels = []
        subs = {}
        for rel in rels:
            # break
            sub_key = ''.join(map(str, [rel['subject']['category']] + rel['subject']['bbox']))
            if sub_key not in subs:
                subs[sub_key] = (len(subs), rel['subject']['category'], rel['subject']['bbox'])
            sub_idx, _, _ = subs[sub_key]
            pred_labels_by_sub[sub_idx].append(rel['predicate'])
            pred_slabels_by_sub[sub_idx].append(pred_label[rel['predicate']])
            box_labels_by_sub[sub_idx].append(rel['object']['category'])
            box_slabels_by_sub[sub_idx].append(obj_label[rel['object']['category']])
            boxes_by_sub[sub_idx].append(rel['object']['bbox'])

        for sub_key in subs:
            sub_labels.append(subs[sub_key][1])
            sub_slabels.append(obj_label[subs[sub_key][1]])
        
        for sub_key, sub_label, sub_slabel in zip(subs, sub_labels, sub_slabels):
            sub_idx, _, _  = subs[sub_key]
            pred_labels = pred_labels_by_sub[sub_idx]
            pred_slabels = pred_slabels_by_sub[sub_idx]
            obj_labels = box_labels_by_sub[sub_idx]
            obj_slabels = box_slabels_by_sub[sub_idx]
            obj_boxes = boxes_by_sub[sub_idx]
            sub_box = subs[sub_key][2]
            row = [uniq_id,
                ','.join(map(str, pred_labels)),
                ','.join(map(str, obj_labels)),
                ','.join([' '.join(map(str, box)) for box in obj_boxes]),
                ','.join(pred_slabels),
                ','.join(obj_slabels),
                sub_label, sub_slabel, ' '.join(map(str, sub_box))]
            # print(row)
            writer.writerow(row)
            line_count += 1

        tqdm_obj.set_postfix({'line_count': line_count / (i + 1)})
        # break