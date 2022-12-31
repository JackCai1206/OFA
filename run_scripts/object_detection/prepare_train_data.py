from os import makedirs, path
from pycocotools import coco

DATA_DIR = '/data/hulab/zcai75/coco'
ANNO_DIR = f'{DATA_DIR}/annotations/instances_val2017.json'
STORE_DIR = '/data/hulab/zcai75/OFA_data'

makedirs(f'{STORE_DIR}/coco_data', exist_ok=True)

coco_anno = coco.COCO(ANNO_DIR)

with open(f"{STORE_DIR}/train.tsv", 'w') as f:
    for img_id in coco_anno.getImgIds():
        ann_ids = coco_anno.getAnnIds(img_id)
        anns = coco_anno.loadAnns(ann_ids)
        label_list = ''
        # image_id, image, label_list(x0, y0, x1, y1, cat_id, cat && ...)
        label_list = '&&'.join([
            ','.join([
                ','.join(str(pt) for pt in ann['bbox']),
                str(ann['category_id']),
                coco_anno.loadCats([ann['category_id']])[0]['name']
            ])
        for ann in anns])
        print(label_list)
        break
    
