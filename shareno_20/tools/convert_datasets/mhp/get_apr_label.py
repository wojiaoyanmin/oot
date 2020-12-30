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

mhp_id2label = {0: 'Background',
                1: 'Cap/hat',
                2: 'Helmet',
                3: 'Face',
                4: 'Hair',
                5: 'Left-arm',
                6: 'Right-arm',
                7: 'Left-hand',
                8: 'Right-hand',
                9: 'Protector',
                10: 'Bikini/bra',
                11: 'Jacket/windbreaker/hoodie ',
                12: 'Tee-shirt',
                13: 'Polo-shirt',
                14: 'Sweater',
                15: 'Singlet',
                16: 'Torso-skin',
                17: 'Pants',
                18: 'Shorts/swim-shorts',
                19: 'Skirt',
                20: 'Stockings',
                21: 'Socks',
                22: 'Left-boot',
                23: 'Right-boot',
                24: 'Left-shoe',
                25: 'Right-shoe',
                26: 'Left-highheel',
                27: 'Right-highheel',
                28: 'Left-sandal',
                29: 'Right-sandal',
                30: 'Left-leg',
                31: 'Right-leg',
                32: 'Left-foot',
                33: 'Right-foot',
                34: 'Coat',
                35: 'Dress',
                36: 'Robe',
                37: 'Jumpsuit',
                38: 'Other-full-body-clothes',
                39: 'Headwear',
                40: 'Backpack',
                41: 'Ball',
                42: 'Bats',
                43: 'Belt',
                44: 'Bottle',
                45: 'Carrybag',
                46: 'Cases',
                47: 'Sunglasses',
                48: 'Eyewear',
                49: 'Glove',
                50: 'Scarf',
                51: 'Umbrella',
                52: 'Wallet/purse',
                53: 'Watch',
                54: 'Wristband',
                55: 'Tie',
                56: 'Other-accessary',
                57: 'Other-upper-body-clothes',
                58: 'Other-lower-body-clothes', }


def collect_files(text_dir,
            instance_dir,
            out_dir):
    
    files = []
    print("collencting files")
    flist = [line.strip() for line in open(text_dir).readlines()]

    for add in tqdm(flist, desc='Loading %s ..' % ('val')):
        file_single=[]
        for instance_file in glob.glob(osp.join(instance_dir, '*.png')):
            instance_name = osp.basename(instance_file)[:-len('.png')]
            instance_index, totoal_num, human_index = instance_name.split('_')
            if instance_index == add:
                num_human=totoal_num
                img = mmcv.imread(instance_file, 'unchanged')[:, :, 2]
                Category_id=numpy.unique(img)
                img[img>0]=( 60*(int(human_index)-1)+img[img>0])
                file_single.append(img)
                for id in Category_id:
                    if id == 0:
                        continue
                    with open(osp.join(out_dir, instance_index  + ".txt"),'a') as f:
                        f.write('%d %d %d\n'%((int(human_index)-1)*60+id,int(id), int(human_index)))
        after_img = numpy.stack(file_single)
        after_img = numpy.max(after_img,axis=0)
        if  numpy.max(after_img)>=int(num_human)*60:
            print('add:',add)
            print('max:',numpy.max(after_img))
            print('total:',int(num_human)*60)
        cv2.imwrite(osp.join(out_dir, add  + ".png"),
                    after_img)
        time.sleep(0.5)
        
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
    text_dir = 'data/MHP/list/val.txt'
    instance_dir = 'data/MHP/val/parsing_annos/'
    out_dir = 'data/MHP/val/instance_part_val'
    mmcv.mkdir_or_exist(out_dir)

    with mmcv.Timer(
            print_tmpl='It tooks {}s to convert MHP annotation'):
        files = collect_files(
            text_dir,
            instance_dir,
            out_dir)


if __name__ == '__main__':
    main()
