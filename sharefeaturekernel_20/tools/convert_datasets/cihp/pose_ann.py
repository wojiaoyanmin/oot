import os
import glob
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2
import time
import pdb
import mmcv
import json
import os.path as osp
import numpy as np
import cv2
def annopicture(json_file, human_dir,category_dir,out_dir):
    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)


    flist = []
    with open(json_file,encoding='utf8')as fp:
        json_data=json.load(fp)
    
    for data in json_data['images']:       
        img_name=osp.split(data['file_name'])[-1][:-4]
        human=cv2.imread(osp.join(human_dir,img_name+'.png'),cv2.IMREAD_GRAYSCALE)
        category = cv2.imread(osp.join(category_dir,img_name+'.png'),cv2.IMREAD_GRAYSCALE)
        num_human=np.max(human)
        for i in range(int(num_human+1)):
            if i==0:
                continue
            new_img=(human==i)*(category)
            cv2.imwrite(osp.join(out_dir,img_name+"_"+str(int(num_human)).zfill(2)+"_"+str(i).zfill(2)+".png"),new_img)
            
        time.sleep(0.5)
        print("done")
        # plt.show()#显示图像


def main():
    #json_file='data/MHP/annotations/Instance_val.json'
    json_file = 'data/CIHP/annotations/Instance_val.json'
    human_dir = 'data/CIHP/val/Human_ids'
    category_dir='data/CIHP/val/Category_ids'
    out_dir = 'data/CIHP/val/parsing_annos'
    mmcv.mkdir_or_exist(out_dir)
    annopicture(json_file, human_dir,category_dir, out_dir)


if __name__ == '__main__':
    main()
