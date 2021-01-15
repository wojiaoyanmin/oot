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
import matplotlib.pyplot as plt
import time
import shutil

import argparse
mhp_id2label = {0: 'Background',
                1: 'head',
                2: 'torso',
                3: 'u-arms',
                4: 'l-arms',
                5: 'u-legs',
                6: 'l-legs',
                }
def get_palette(num_cls):

    color=[[0,     0,     0],[128,     0,     0],[ 0,   128,     0],[128,   128,     0],[0,     0 ,  128],[128  ,   0 ,  128],[0  , 128,   128]]
    inds=[0,4,2,6,1,5,3]

    return color,inds

def collect_files(text_dir,
            img_dir,
            out_dir):
    
    files = []
    print("collencting files")
    flist = [line.strip() for line in open(text_dir).readlines()]
    for add in tqdm(flist, desc='Loading %s ..' % ('val')):
       img=osp.join(img_dir,add+'.jpg')
       shutil.copy(osp.join(img_dir,add+'.jpg'),osp.join(out_dir,add+'.jpg'))
        
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert mhp annotations to COCO format')
    #parser.add_argument('mhp_path', help='mhp data path')
    parser.add_argument('--Images', default='images', type=str)
    parser.add_argument('--Categoriy-dir', default='Category_ids', type=str)
    parser.add_argument('--Human-dir', default='Human_ids', type=str)
    parser.add_argument('--Instance-dir', default='parsing_annos', type=str)
    #parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    img_dir = 'data/pascal/JPEGImages'
    text_dir = 'data/pascal/list/val_id.txt'
    human_dir = 'data/pascal/Human_ids/'
    cateory_dir = 'data/pascal/Categories'
    out_dir = 'data/pascal/val/Images'
    mmcv.mkdir_or_exist(out_dir)

    with mmcv.Timer(
            print_tmpl='It tooks {}s to convert MHP annotation'):
        files = collect_files(
            text_dir,
            img_dir,
            out_dir)


if __name__ == '__main__':
    main()
