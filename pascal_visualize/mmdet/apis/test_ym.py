import os.path as osp
import pickle
import shutil
import tempfile
import time
from PIL import Image as PILImage
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import numpy as np
from mmdet.core import encode_mask_results, tensor2imgs, get_classes
import pdb
import gc
import pycocotools.mask as mask_util
from scipy import ndimage
import cv2
import pdb
import os
import scipy.sparse
import pickle,gzip
import mmcv
import matplotlib.pyplot as plt
import scipy.io as scio
# def get_palette():
#     datafile = "data/MHP/LV-MHP-v2_colormap.mat"
#     data = (scio.loadmat(datafile)['MHP_colormap'].flatten() * 255).astype(int).tolist()
#     return data

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

def get_masks(result, num_classes=8):
    cur_result = result
        #ins
    encode = [[] for _ in range(num_classes)]
    if cur_result[0] is None:
        return encode
    ins_mask = cur_result[0].cpu().numpy().astype(np.uint8)
    ins_label = cur_result[1].cpu().numpy().astype(np.int)
    ins_score = cur_result[2].cpu().numpy().astype(np.float)
    assert len(ins_mask)==len(ins_label)==len(ins_score),'dimension wrong'
    num_ins = ins_mask.shape[0]
    for idx in range(num_ins):
        cur_mask = ins_mask[idx, ...]
        rle = mask_util.encode(
            np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
        rst = (rle, ins_score[idx])
        encode[ins_label[idx]].append(rst)

    return encode

def get_semantic(result,num_classes=8):
    for cur_result in result:
        encode = []
        if cur_result[0] is None:
            return encode
        ins_mask = cur_result[0].cpu().numpy().astype(np.uint8)
        num_ins = ins_mask.shape[0]
        for idx in range(num_ins):
            cur_mask = ins_mask[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            encode.append(rle)
    return encode

def vis_seg(data, result, img_norm_cfg, score_thr, save_dir):
    result=[result]
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    class_names = get_classes('PAS')
    palette = get_palette(len(class_names)+10)
    
    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        if cur_result[0] is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        seg_label = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy()
        score = cur_result[2].cpu().numpy()
        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]  # 从小到大排
        seg_show = img_show.copy()
        seg_show=np.zeros_like(img_show)
        seg_show 
        for idx in range(num_mask):
            idx = -(idx + 1)
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue

            # color_mask = np.random.randint(
            #     0, 256, (1, 3), dtype=np.uint8)

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]
            
            color_mask = palette[int(cur_cate*3):int(cur_cate*3+3)]
            color_mask=np.array(color_mask)
            cur_mask_bool = cur_mask.astype(np.bool)
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0 + color_mask * 1
            label_text = class_names[int(cur_cate)]
            # label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (int(center_x), int(center_y))
            # cv2.putText(seg_show, label_text, vis_pos,
            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))  


        
            data_id=os.path.split(img_metas[0]['filename'])[-1]
            mmcv.mkdir_or_exist(save_dir)
            '''seg_show = PILImage.fromarray(seg_show)
            seg_show.save(os.path.join(save_dir,data_id))'''
            mmcv.imwrite(seg_show, os.path.join(save_dir,data_id))
            # print(cur_score)
            # print(color_mask)
            # pdb.set_trace()


def single_gpu_test(model,
                    data_loader,
                    show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    num_classes = len(dataset.CLASSES)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=True, **data)
        result = get_masks(seg_result, num_classes=num_classes)
        results.append(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

        batch_size = (
            len(data['img_meta']._data)
            if 'img_meta' in data else data['img'][0].size(0))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, args, cfg, tmpdir=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()

    results = []
    instance_results = []
    dataset = data_loader.dataset
    num_classes = len(dataset.CLASSES)
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(num_classes)]

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    cache='tmp/'
    apr_output_dirs='apr/'

    # if os.path.exists('tmp'):
    #     shutil.rmtree('tmp')
    # if os.path.exists('apr'):
    #     shutil.rmtree('apr')
    # mmcv.mkdir_or_exist(apr_output_dirs)
    # mmcv.mkdir_or_exist(cache)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            pred_result = model(return_loss=False, rescale=True, **data)[0]

            key = pred_result[0]
            results_cache_add = cache + key + '.pklz'
            apr =pred_result[1]['INSTANCE']
            pred_result[1].pop('INSTANCE')
            #app
            # app=pred_result[1]
            # pickle.dump(app, gzip.open(results_cache_add, 'w'))
            # #apr
            # instance_seg_masks,instance_cate_labels,instance_cate_scores = apr
            # if instance_seg_masks is None:
            #     continue 
            # results_apr_add = apr_output_dirs + key +'.pklz'
            # pickle.dump(instance_seg_masks, gzip.open(results_apr_add, 'w'))
            # for label,score in zip(instance_cate_labels, instance_cate_scores):
            #     with open(os.path.join(apr_output_dirs, '%s.txt' % key), 'a') as f:
            #         f.write('%d %f\n' % (label+1, score))
            encoder_ins = get_masks(apr, num_classes=num_classes)
            # instance_results.append(encoder_ins)       
            results.append(encoder_ins)
        if args.show:
            vis_seg(data, apr, cfg.img_norm_cfg,
                    score_thr=args.show_score_thr, save_dir=args.show_dir)
        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))

            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)
    
    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results



