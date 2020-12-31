import argparse
import glob
import os.path as osp
import pdb
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import argparse
import glob
import os.path as osp
import pdb
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from tqdm import trange, tqdm
import numpy
import cv2
import utils
import matplotlib.pyplot as plt
import time
import os
import os.path as osp
cihp_id2label={0: 'Background',
                1: 'head',
                2: 'torso',
                3: 'u-arms',
                4: 'l-arms',
                5: 'u-legs',
                6: 'l-legs',
                }
train_label =['head','torso','left-u-arms','left-l-arms','left-u-legs','left-l-legs','right-u-arms','right-l-arms','right-u-legs','right-l-legs']
detailpart2partlabel={'':'',

}
def collect_files(text_dir,Images_dir, Instance_dir):
    files = []
    suffix='jpg'
    flist = [line.strip() for line in open(text_dir).readlines()]
    for add in tqdm(flist, desc='Loading %s ..' % ('val')):
        Image_file = osp.join(Images_dir, add+'.jpg')
        Instance_file = osp.join(Instance_dir, add+'.mat')
        files.append((Image_file, Instance_file))
    assert len(files), f'No images found in {Images_dir}'
    print(f'Loaded {len(files)} images from {Images_dir}')
    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)
   
    return images


def load_img_info(files):
    Image_file, Instance_file= files

    annotations = utils.load_annotations(Instance_file)
    obj_cnt = 0

    # Show original image with its mask:
    for obj in annotations["objects"]:
        if obj["class"] == "person":
            img_list = {value:np.zeros_like(obj['mask']) for key,value in enumerate(train_label)}
            obj_cnt = obj_cnt + 1
            for part_id in detailpart2partlabel:
                if detailpart2partlabel[part_id] in train_label:
                    img_list[detailpart2partlabel[part_id]]
            mask = np.asarray(Instance_img == id, dtype=np.uint8, order='F')
        else:
            continue



    Instance_img = mmcv.imread(Instance_file, 'unchanged')
    ids = np.unique(Instance_img)
    anno_info = []
    for id in ids:
        if id == 0:
            continue
        category_id = int(id)%len(cihp_id2label)-1  # 把background作为最后一个类别
        instance_id= int(id)//len(cihp_id2label)
        iscrowd = 0
        mask = np.asarray(Instance_img == id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]
        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)
        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()
        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            instance_id=instance_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(Image_file))
    info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(Image_file)),
        height=Instance_img.shape[0],
        width=Instance_img.shape[1],
        anno_info=anno_info,
        segm_file=None)
    return info

def cvt_annotations(infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for info in infos:
        info['id'] = img_id
        anno_infos = info.pop('anno_info')
        out_json['images'].append(info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    for label, name in cihp_id2label.items():
        if label==0:
            continue
        cat = dict(id=label-1, name=name)
        out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')
    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert cihp annotations to COCO format')
    parser.add_argument('cihp_path', help='cihp data path')
    parser.add_argument('--Images', default='Images', type=str)
    parser.add_argument('--Categoriy-dir', default='Category_ids', type=str)
    parser.add_argument('--Human-dir', default='Human_ids', type=str)
    parser.add_argument('--Instance-dir', default='Instance_ids', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    Images_dir='data/PAS/JPEGImages'
    Instance_dir='data/PAS/train/Annotations_Part'
    text_dir = 'data/PAS/train/train_id.txt'
    out_dir='data/PAS/annotations'
    mmcv.mkdir_or_exist(out_dir)


    set_name = dict(
        val='Instance_train.json'
        )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert Cityscapes annotation'):
            files = collect_files(
                text_dir,
                osp.join(Images_dir),
                osp.join(Instance_dir))
            infos = collect_annotations(files, nproc=1)
            cvt_annotations(infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
