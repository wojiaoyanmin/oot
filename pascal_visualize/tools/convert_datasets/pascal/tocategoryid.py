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
            human_dir,
            cateory_dir,
            out_dir):
    
    files = []
    print("collencting files")
    flist = [line.strip() for line in open(text_dir).readlines()]
    color,inds=get_palette(6)
    for add in tqdm(flist, desc='Loading %s ..' % ('val')):
        human_name=osp.join(human_dir,add+'.png')
        category_name=osp.join(cateory_dir,add+'.png')
        
        category_img = mmcv.imread(category_name,'unchanged')
        instance_img = np.zeros_like(category_img[:,:,0])
        (hh,ww,cc)=category_img.shape
        for color_id,ind in zip(color,inds):
            aa=(category_img[:,:,0]==color_id[0])&(category_img[:,:,1]==color_id[1])&(category_img[:,:,2]==color_id[2])
            instance_img[aa]=ind

        
            # for i in range(hh):
            #     for j in range(ww):
            #         if (category_img[i,j][0]==color_id[0])&(category_img[i,j][1]==color_id[1])&(category_img[i,j][2]==color_id[2]):

        cv2.imwrite(osp.join(out_dir, add  + ".png"),instance_img)


        # Category_id=numpy.unique(img)
        # img[img>0]=( 60*(int(human_index)-1)+img[img>0])
        # file_single.append(img)
        # for id in Category_id:
        #     if id == 0:
        #         continue
        #     with open(osp.join(out_dir, instance_index  + ".txt"),'a') as f:
        #         f.write('%d %d %d\n'%((int(human_index)-1)*60+id,int(id), int(human_index)))
        # after_img = numpy.stack(file_single)
        # after_img = numpy.max(after_img,axis=0)
        # if  numpy.max(after_img)>=int(num_human)*60:
        #     print('add:',add)
        #     print('max:',numpy.max(after_img))
        #     print('total:',int(num_human)*60)
        # cv2.imwrite(osp.join(out_dir, add  + ".png"),
        #             after_img)
        # time.sleep(0.5)
        
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
    text_dir = 'data/pascal/list/train_id.txt'
    human_dir = 'data/pascal/Human_ids/'
    cateory_dir = 'data/pascal/Categories'
    out_dir = 'data/pascal/Categories_ids'
    mmcv.mkdir_or_exist(out_dir)

    with mmcv.Timer(
            print_tmpl='It tooks {}s to convert MHP annotation'):
        files = collect_files(
            text_dir,
            human_dir,
            cateory_dir,
            out_dir)


if __name__ == '__main__':
    main()
