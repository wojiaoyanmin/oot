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
import scipy.io

cihp_id2label={0: 'Background',
                1: 'head',
                2: 'torso',
                3: 'u-arms',
                4: 'l-arms',
                5: 'u-legs',
                6: 'l-legs',
                }
train_label ={'head':1,'torso':2,'left-u-arms':3,'left-l-arms':4,'left-u-legs':5,'left-l-legs':6,'right-u-arms':3,'right-l-arms':4,'right-u-legs':5,'right-l-legs':6}
detailpart2partlabel={'ruleg':'right-u-legs',
                        'rlleg':'right-l-legs',
                        'ruarm':'right-u-arms',
                        'rlarm':'right-l-arms',
                        'luleg':'left-u-legs',
                        'llleg':'left-l-legs',
                        'luarm':'left-u-arms',
                        'llarm':'left-l-arms',
                        'neck':'head',
                        'torso':'torso',
                        'nose':'head',
                        'hair':'head',
                        'mouth':'head',
                        'lebrow':'head',
                        'rebrow':'head',
                        'lear':'head',
                        'leye':'head',
                        'reye':'head',
                        'rear':'head',
                        'head':'head',
                        'rhand':'right-l-arms',
                        'lhand':'left-l-arms',
                        'lfoot':'left-l-legs',
                        'rfoot':'right-l-legs'
}


# Load annotations from .mat files creating a Python dictionary:
def load_annotations(path):

    # Get annotations from the file and relative objects:
    annotations = scipy.io.loadmat(path)["anno"]

    objects = annotations[0, 0]["objects"]

    # List containing information of each object (to add to dictionary):
    objects_list = []

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]
        mask = obj["mask"]

        # List containing information of each body part (to add to dictionary):
        parts_list = []

        parts = obj["parts"]

        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_mask = part["mask"]

            # Add info to parts_list:
            parts_list.append({"part_name": part_name, "mask": part_mask})

        # Add info to objects_list:
        objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

    return {"objects": objects_list}
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
    # images=load_img_info(files[3])
    return images


def load_img_info(files):
    Image_file, Instance_file= files

    annotations = load_annotations(Instance_file)
    anno_info = []
    zero_img=np.zeros_like(annotations["objects"][0]['mask'])
    cnt=0
    # Show original image with its mask:
    for obj in annotations["objects"]:
        
        if obj["class"] == "person":
            cnt=cnt+1
            
            img_list={}
            for label in train_label.keys():
                img_list[label]=zero_img
            
            for part_id in obj['parts']:
                small_part_name=part_id['part_name']
                small_part_mask=part_id['mask']
                big_part_name= detailpart2partlabel[small_part_name]

                img_list[big_part_name]=(small_part_mask+img_list[big_part_name]).astype(np.bool)
            for big_part_name,mask in img_list.items():
                if mask.sum()<=0:
                    continue
                else:
                     mask = np.asarray(mask, dtype=np.uint8, order='F')
                     iscrowd = 0
                     mask_rle = maskUtils.encode(mask[:, :, None])[0]
                     area = maskUtils.area(mask_rle)
                    # convert to COCO style XYWH format
                     bbox = maskUtils.toBbox(mask_rle)
                    # for json encoding
                     mask_rle['counts'] = mask_rle['counts'].decode()
                     anno = dict(
                        iscrowd=iscrowd,
                        category_id=train_label[big_part_name]-1,
                        instance_id=cnt,
                        bbox=bbox.tolist(),
                        area=area.tolist(),
                        segmentation=mask_rle)
                     anno_info.append(anno)
    video_name = osp.basename(osp.dirname(Image_file))
    info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(Image_file)),
        height=zero_img.shape[0],
        width=zero_img.shape[1],
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
