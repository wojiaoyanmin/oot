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
import numpy as np
import pycocotools.mask as mask_util
from mmdet.core import encode_mask_results, tensor2imgs, get_classes
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    '''
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    '''
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 2] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 0] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    return palette[3:]


def jsontopicture(json_file, dataset_dir, out_dir):
    class_names = get_classes('MHP')
    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # json_file='../../datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
    # dataset_dir='data/cityscapes/leftImg8bit/val/'
    coco = COCO(json_file)
    # catIds=coco.getCatIds(catNms=['person'])#catIds=1表示人这一类
    imgIds = coco.getImgIds()  # 图片id，许多值
    palette = get_palette(100)
    aa=0
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]

        I = cv2.imread(dataset_dir + img['file_name'])
        seg_show=np.zeros_like(I)
        # plt.show()
        

        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for j in range(len(anns)):
            cur_cate = anns[j]['category_id']
            #cur_instance = anns[j]['instance_id']

            cur_mask = mask_util.decode(anns[j]['segmentation'])
            # color_mask = palette[int(cur_instance*3):int(cur_instance*3+3)]
            # color_mask=np.array(color_mask)
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            label_text = class_names[int(cur_cate)]
            cur_mask_bool = cur_mask.astype(np.bool)

            seg_show[cur_mask_bool] = color_mask * 1
        
        
        subfile = os.path.split(img['file_name'])
        path = os.path.join(out_dir, '{}'.format(subfile[0]))
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, '{}'.format(subfile[1])),seg_show)
        aa=aa+1
        #plt.savefig(os.path.join(path, '{}'.format(subfile[1])))
        time.sleep(0.5)
        print(aa)
        # plt.show()#显示图像


def main():
    json_file='data/MHP/annotations/Instance_trybig.json'
    #json_file = 'data/CIHP/annotations/Instance_val.json'
    dataset_dir = 'data/MHP/val/'
    out_dir = 'work_dirs/gtvis/'
    mmcv.mkdir_or_exist(out_dir)
    jsontopicture(json_file, dataset_dir, out_dir)


if __name__ == '__main__':
    main()
