import base64
from io import BytesIO
from os import makedirs, path
from pycocotools import coco
from csv import writer
from PIL import Image
from tqdm import tqdm

DATA_DIR = '/data/hulab/zcai75/coco'
STORE_DIR = '/data/hulab/zcai75/OFA_data'

makedirs(f'{STORE_DIR}/coco_data', exist_ok=True)

def xywh2xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

for split in ['train', 'val']:
    skipped = 0
    coco_anno = coco.COCO(f'{DATA_DIR}/annotations/instances_{split}2017.json')
    with open(f"{STORE_DIR}/{split}.tsv", 'w') as f:
        wtr = writer(f, delimiter='\t')
        for img_id in tqdm(coco_anno.getImgIds()[0:1000]):
            ann_ids = coco_anno.getAnnIds(img_id)
            if len(ann_ids) == 0:
                skipped += 1
                continue
            anns = coco_anno.loadAnns(ann_ids)
            label_list = '&&'.join([
                ','.join([
                    ','.join(str(pt) for pt in xywh2xyxy(ann['bbox'])),
                    str(ann['category_id']),
                    coco_anno.loadCats([ann['category_id']])[0]['name']
                ])
            for ann in anns])
            with open(f"{DATA_DIR}/images/{split}2017/{coco_anno.loadImgs([img_id])[0]['file_name']}", 'rb') as image:
                # image_id, image, label_list(x0, y0, x1, y1, cat_id, cat && ...)
                img_str = base64.urlsafe_b64encode(image.read()).decode('utf-8')
                wtr.writerow([img_id, img_str, label_list])
    print(f'skipped: {skipped}')

    
