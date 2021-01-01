#coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch
import pdb
import os.path as osp
#from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    heatmap = torch.mean(feature_map,dim=0)
    heatmap = heatmap.cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def draw_feature_map(featuremap, img_path, save_dir='work_dirs/visualize'):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = mmcv.imread(img_path)
    heatmap = featuremap_2_heatmap(featuremap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.5 + img*0.3  # 这里的0.4是热力图强度因子
    # cv2.imshow("1",superimposed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(osp.join(save_dir,osp.split(img_path)[-1][:-4]+".png"), superimposed_img)  # 将图像保存到硬盘

    # show_result_pyplot(model, img, result, score_thr=0.05)