import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule
from mmdet.core import multi_apply, bbox2roi, matrix_nms, multiclass_nms
from ..builder import HEADS, build_loss,build_head
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt
from mmdet.ops import ConvDW
import os.path as osp
INF = 1e8
import numpy as np
import scipy.sparse
from mmdet.core import BitmapMasks
import pycocotools.mask as mask_util
from skimage.measure import label

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep



@HEADS.register_module
class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_human_grids=None,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_cate=None,
                 human_loss_ins=None,
                 human_loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.num_human_grids = num_human_grids
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes 
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.scale_ranges = scale_ranges
        self.loss_cate = build_loss(loss_cate)
        self.loss_ins= build_loss(loss_ins)
        self.human_loss_cate = build_loss(human_loss_cate)
        self.human_loss_ins = build_loss(human_loss_ins)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.human_cate_convs = nn.ModuleList()
        self.human_kernel_convs = nn.ModuleList()
        cfg_conv = self.conv_cfg
        norm_cfg = self.norm_cfg


        for i in range(self.stacked_convs):
            
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            self.human_kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            self.human_cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

        self.human_solo_cate = nn.Conv2d(
            self.seg_feat_channels, 1, 3, padding=1)

        self.human_solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        for m in self.human_cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.human_kernel_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)
        normal_init(self.human_solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.human_solo_kernel, std=0.01)

    def forward(self, feats,img_metas=None, eval=False):
        '''
        print(img_metas[0]['filename'])
        filename=img_metas[0]['filename']
        filename=osp.join(osp.join(osp.dirname(osp.split(filename)[0]),'Instances'),osp.split(filename)[1][:-3]+'png')
        img=plt.imread(filename)
        showimg=[]
        for feat in feats:
            showimg.append(F.interpolate(feat,size=feats[0].size()[-2:],mode='bilinear'))
        
        plt.subplot(2,3,6)
        plt.imshow(img)
        plt.subplot(2,3,1)
        plt.imshow(torch.max(showimg[0][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,2)
        plt.imshow(torch.max(showimg[1][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,3)
        plt.imshow(torch.max(showimg[2][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,4)
        plt.imshow(torch.max(showimg[3][0],0)[0].detach().cpu().numpy())
        plt.show()
        '''

        human_feats = feats[-1]
        parts_feats = feats[:-1]
        parts_feats = self.split_feats(parts_feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in parts_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, parts_feats,
                                                        list(range(len(self.seg_num_grids))),
                                                        img_metas=img_metas,
                                                        eval=eval, upsampled_size=upsampled_size)
        
        x_range = torch.linspace(-1, 1, human_feats.shape[-1], device=human_feats.device)
        y_range = torch.linspace(-1, 1, human_feats.shape[-2], device=human_feats.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([human_feats.shape[0], 1, -1, -1])
        x = x.expand([human_feats.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        human_feats = torch.cat([human_feats, coord_feat], 1)

        human_feats = F.interpolate(human_feats, size=self.num_human_grids, mode='bilinear', align_corners=True)
        feats_human_kernel = human_feats
        feats_human_cate = human_feats[:,:-2,:,:].contiguous()
        for conv in self.human_cate_convs:
            feats_human_cate = conv(feats_human_cate)
        feats_human_cate = feats_human_cate.contiguous()
        for conv in self.kernel_convs:
            feats_human_kernel = conv(feats_human_kernel)
        human_cate_pred = self.human_solo_cate(feats_human_cate)
        human_kernel_pred = self.solo_kernel(feats_human_kernel)
        #human_kernel_pred = F.softmax(human_kernel_pred,dim=1)
        
        if eval:
            human_cate_pred = points_nms(human_cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred,human_cate_pred, human_kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                F.interpolate(feats[3], scale_factor=2, mode='bilinear'))

    def forward_single(self, x, idx, img_metas=None, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        #kernel_pred = F.softmax(kernel_pred,dim=1)
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def loss(self,
             cate_preds,
             kernel_preds,
             human_cate_pred,
             human_kernel_pred,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             gt_instance_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        mask_feat_size = ins_pred.size()[-2:]
        #parts
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]

        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.reshape(N,-1)
                cur_ins_pred = torch.matmul(kernel_pred.permute(1,0),cur_ins_pred).reshape(I,H,W)

                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()
        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(self.loss_ins(input, target).unsqueeze(0))
        loss_ins = torch.cat(loss_ins).mean()
        # cate

        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        '''
        weight=torch.Tensor([0.9781,0.9557,1.0731,1.0411,0.9562,0.9897,\
            0.963,1.0086,0.9588,0.9588,1.0607,1.0537,0.9556,0.9739,0.9735,\
            1.0328,1.0328,1.0184,1.0186]).expand(flatten_cate_preds.shape).to(flatten_cate_preds.device)'''
        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)

        #!!!!!!!!!human
        #human
        human_ins_label, human_cate_label, human_ins_ind_label, human_grid_order = multi_apply(
            self.human_single,
            gt_instance_list,
            gt_bbox_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)
        human_ins_labels = torch.cat([human_ins_label_img for human_ins_label_img in human_ins_label], 0)

        human_kernel_preds = [human_kernel_pred_img.view(human_kernel_pred_img.shape[0], -1)[:, human_grid_order_img]
                         for human_kernel_pred_img, human_grid_order_img in
                         zip(human_kernel_pred, human_grid_order)]
        # generate human masks
        human_ins_pred = ins_pred
        human_mask_pred = []
        for idx, kernel_pred in enumerate(human_kernel_preds):
            if kernel_pred.size()[-1] == 0:
                continue
            cur_ins_pred = human_ins_pred[idx, ...]
            H, W = cur_ins_pred.shape[-2:]
            N, I = kernel_pred.shape
            cur_ins_pred = cur_ins_pred.reshape(N,-1)
            cur_ins_pred = torch.matmul(kernel_pred.permute(1,0),cur_ins_pred).reshape(I,H,W)
            human_mask_pred.append(cur_ins_pred)
        if len(human_mask_pred) == 0:
            human_mask_pred = None
        else:
            human_mask_pred = torch.cat(human_mask_pred, 0)
        
        human_ins_ind_labels =torch.cat([human_ins_ind_label_img.flatten()
                       for human_ins_ind_label_img in human_ins_ind_label])
        human_num_ins = human_ins_ind_labels.sum()
        # dice loss
        human_mask_pred = torch.sigmoid(human_mask_pred)
        human_loss_ins = self.human_loss_ins(human_mask_pred, human_ins_labels)
        # for i in range(human_ins_labels.shape[0]):
        #     plt.imshow(human_ins_labels[i].cpu().numpy())
        #     plt.show()

        # cate
        human_cate_label =  torch.cat([human_cate_label_img.flatten()
                       for human_cate_label_img in human_cate_label])

        human_cate_pred = human_cate_pred.permute(0, 2, 3, 1).reshape(-1, 1)
        human_loss_cate = self.human_loss_cate(human_cate_pred, human_cate_label, avg_factor=human_num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate,
            human_loss_ins = human_loss_ins,
            human_loss_cate=human_loss_cate
        )

    def human_single(self,
                     gt_instance_raw,
                     gt_bboxes_raw,
                     gt_masks_raw,
                     mask_feat_size):
        device=gt_bboxes_raw[0].device
        gt_masks_human = []
        gt_labels_human = []
        gt_bboxes_human = []
        height = gt_masks_raw.height
        width = gt_masks_raw.width
        human_ids = torch.unique(gt_instance_raw)
        for ids in human_ids:

            keep = (gt_instance_raw == ids)
            gt_instance = gt_instance_raw[keep.cpu().numpy()]
            gt_bboxes = gt_bboxes_raw[keep.cpu().numpy()]
            gt_masks = gt_masks_raw[keep.cpu().numpy()]

            left = torch.min(gt_bboxes[:, 0])
            top = torch.min(gt_bboxes[:, 1])
            right = torch.max(gt_bboxes[:, 2])
            bottom = torch.max(gt_bboxes[:, 3])
            if ((right - left) <= 4) or ((bottom - top) <= 4):
                continue
            gt_masks = gt_masks.to_ndarray()
            gt_masks = np.sum(gt_masks, axis=0).astype(np.uint8)
            gt_masks = torch.from_numpy(gt_masks).to(device)
            gt_masks_human.append(gt_masks.unsqueeze(0))
            gt_labels_human.append(torch.tensor([0],dtype=torch.int64,device=device))
            gt_bboxes_human.append(torch.tensor([left,top,right,bottom],device=device).unsqueeze(0))
        
        gt_labels_human=torch.cat(gt_labels_human)
        gt_bboxes_human = torch.cat(gt_bboxes_human)
        gt_masks_human = torch.cat(gt_masks_human).cpu().numpy()
        gt_masks_human = BitmapMasks(gt_masks_human, height, width)
        num_ins=len(gt_masks_human)

        ins_label = []
        grid_order = []
        cate_label = torch.ones([self.num_human_grids, self.num_human_grids], dtype=torch.int64, device=device)
        ins_ind_label = torch.zeros([self.num_human_grids ** 2], dtype=torch.bool, device=device)

        if num_ins == 0:
            ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            return ins_label, cate_label, ins_ind_label, grid_order

        half_ws = 0.5 * (gt_bboxes_human[:, 2] - gt_bboxes_human[:, 0]) * self.sigma
        half_hs = 0.5 * (gt_bboxes_human[:, 3] - gt_bboxes_human[:, 1]) * self.sigma

        output_stride = 4
        for seg_mask, gt_label, half_h, half_w in zip(gt_masks_human, gt_labels_human, half_hs, half_ws):
            if seg_mask.sum() == 0:
                continue
            # mass center
            upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
            center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
            coord_w = int((center_w / upsampled_size[1]) // (1. / self.num_human_grids))
            coord_h = int((center_h / upsampled_size[0]) // (1. / self.num_human_grids))

            # left, top, right, down
            top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / self.num_human_grids)))
            down_box = min(self.num_human_grids - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / self.num_human_grids)))
            left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / self.num_human_grids)))
            right_box = min(self.num_human_grids - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / self.num_human_grids)))

            top = max(top_box, coord_h - 1)
            down = min(down_box, coord_h + 1)
            left = max(coord_w - 1, left_box)
            right = min(right_box, coord_w + 1)

            cate_label[top:(down + 1), left:(right + 1)] = gt_label
            seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
            seg_mask = torch.Tensor(seg_mask)
            for i in range(top, down + 1):
                for j in range(left, right + 1):
                    label = int(i * self.num_human_grids + j)

                    cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                device=device)
                    cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                    ins_label.append(cur_ins_label)
                    ins_ind_label[label] = True
                    grid_order.append(label)
        ins_label = torch.stack(ins_label, 0)

        return ins_label, cate_label, ins_ind_label, grid_order

    def solov2_target_single(self,
                            gt_bboxes_raw,
                            gt_labels_raw,
                            gt_masks_raw,
                            mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        
        # for cat,mask in zip(gt_labels_raw,gt_masks_raw):
        #     print(cat)
        #     plt.imshow(mask)
        #     plt.show()
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()

            num_ins = len(hit_indices)
            ins_label = []
            grid_order = []
            cate_label = torch.ones([num_grid, num_grid], dtype=torch.int64, device=device)*self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() == 0:
                   continue
                # mass center
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            ins_label = torch.stack(ins_label, 0)

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_seg(self, cate_preds, kernel_preds, human_cate_pred,
             human_kernel_pred,seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                for i in range(num_levels)
            ]

            human_cate = human_cate_pred[img_id].view(-1, 1).detach()
            human_kernel = human_kernel_pred[img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            img_name = img_metas[img_id]['filename']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result,_ = self.get_seg_single(cate_pred_list, kernel_pred_list,seg_pred_list,
                                        human_cate, human_kernel,
                                         featmap_size, img_shape, ori_shape,img_name, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       kernel_preds,
                       ins_preds,
                       human_cate_preds, 
                       human_kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       img_name,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        assert len(cate_preds) == len(kernel_preds)
        img_name=osp.split(img_name)[-1][:-4]
        None_results={}
        None_results['MASKS']=[]
        None_results['DETS']=[]
        None_results['INSTANCE']=(None,None,None)
        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        #!!!!!!!!parts
        # process.
        inds = (cate_preds > cfg.score_thr)&(cate_preds==(torch.max(cate_preds,dim=1,keepdim=True))[0])
        # inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return (img_name,None_results) , None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        H, W = ins_preds.shape[-2:]
        I,N = kernel_preds.shape
        ins_preds = ins_preds.reshape(N,-1)
        
        seg_preds = torch.matmul(kernel_preds,ins_preds).reshape(I,H,W).sigmoid()

        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return (img_name,None_results) , None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks

        cate_scores *= seg_scores

        sort_inds = torch.argsort(cate_scores, descending=False)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[-cfg.max_per_img:]
        seg_preds = seg_preds[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        sum_masks = sum_masks[sort_inds]
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_preds = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks=seg_preds>cfg.mask_thr

        # sort_inds = torch.argsort(cate_scores, descending=True)
        # if len(sort_inds) > cfg.nms_pre:
        #     sort_inds = sort_inds[:cfg.nms_pre]
        # seg_preds = seg_preds[sort_inds, :, :]
        # seg_masks = seg_masks[sort_inds]
        # cate_scores = cate_scores[sort_inds]
        # cate_labels = cate_labels[sort_inds]
        # sum_masks = sum_masks[sort_inds]
        # cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
        #                          kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)
        # #filter.parts

        # keep = cate_scores >= cfg.update_thr
        # if keep.sum() == 0:
        #     return [None]*2
        
        # # sort and keep top_k
        # seg_preds = seg_preds[keep,...]
        # seg_masks = seg_masks[keep,...]
        # cate_scores = cate_scores[keep]
        # cate_labels = cate_labels[keep]



        #human
        # process.
        
        human_inds = (human_cate_preds > cfg.human_score_thr)
        human_cate_scores = human_cate_preds[human_inds]
        if len(human_cate_scores) == 0:
            return (img_name,None_results) , None
        # cate_labels & kernel_preds
        human_inds = human_inds.nonzero()
        human_cate_labels = human_inds[:, 1]
        human_kernel_preds = human_kernel_preds[human_inds[:, 0]]
        
        # mask encoding.
        I, N = human_kernel_preds.shape
        ins_preds = ins_preds.reshape(N,-1)
        
        human_seg_preds = torch.matmul(human_kernel_preds,ins_preds).reshape(I,H,W).sigmoid()

        # mask.
        human_seg_masks = human_seg_preds > cfg.mask_thr
        human_sum_masks = human_seg_masks.sum((1, 2)).float()

        # filter.
        human_keep = human_sum_masks > 10
        if human_keep.sum() == 0:
            return (img_name,None_results) , None

        human_seg_masks = human_seg_masks[human_keep, ...]
        human_seg_preds = human_seg_preds[human_keep, ...]
        human_sum_masks = human_sum_masks[human_keep]
        human_cate_scores = human_cate_scores[human_keep]
        human_cate_labels = human_cate_labels[human_keep]

        # mask scoring.
        human_seg_scores = (human_seg_preds * human_seg_masks.float()).sum((1, 2)) / human_sum_masks

        human_cate_scores *= human_seg_scores


        human_sort_inds = torch.argsort(human_cate_scores, descending=True)
        human_seg_masks = human_seg_masks[human_sort_inds, :, :]
        human_seg_preds = human_seg_preds[human_sort_inds, :, :]
        human_sum_masks = human_sum_masks[human_sort_inds]
        human_cate_scores = human_cate_scores[human_sort_inds]
        human_cate_labels = human_cate_labels[human_sort_inds]

        human_cate_scores = matrix_nms(human_seg_masks, human_cate_labels, human_cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=human_sum_masks)
        #filter.parts
        
        human_keep = human_cate_scores >= cfg.human_update_thr
        if human_keep.sum() == 0:
            return (img_name,None_results) , None
        
        # sort and keep top_k
        human_seg_preds = human_seg_preds[human_keep,...]
        human_seg_masks = human_seg_masks[human_keep,...]
        human_cate_scores = human_cate_scores[human_keep]
        human_cate_labels = human_cate_labels[human_keep]

        human_seg_preds = F.interpolate(human_seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        human_seg_preds = F.interpolate(human_seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        human_seg_masks=human_seg_preds>cfg.mask_thr

        #small to large
        # sem_pred=torch.zeros([self.num_classes+1, human_seg_masks.shape[-2] , seg_masks.shape[-1]], dtype=torch.float, device=human_seg_masks.device)
        # for cate_score,cate_label,seg_mask in zip(cate_scores,cate_labels,seg_masks):
        #     sem_pred[cate_label+1][seg_mask]=cate_score
        
        results={}
        mask_list=[]
        det_list=[]
        instance_seg_masks=[]
        instance_cate_labels=[]
        instance_cate_scores = []
        num_humans=human_seg_masks.shape[0]
        num_parts = seg_masks.shape[0]
        inter_matrix=torch.mm(human_seg_masks.float().reshape(num_humans,-1),seg_masks.float().reshape(num_parts,-1).permute(1,0))#num_human*num_parts
        parts_matrix = seg_masks.sum((-2,-1)).unsqueeze(0).expand(num_humans,num_parts)
        ratio = inter_matrix/parts_matrix
        keep_parts = (ratio>cfg.assign_parts_th)

        

        for i in range(num_humans):
            # plt.imshow(human_seg_masks[i].cpu().numpy())
            # print(img_name)
            # print('before')
            # plt.show()
            keep=keep_parts[i]
            if keep.sum()==0:
                continue
            part_pred=seg_preds[keep]
            part_mask=seg_masks[keep]
            part_score = cate_scores[keep]
            part_label = cate_labels[keep]
            # for i in range(part_pred.shape[0]):
            #     plt.imshow(part_pred[i].cpu().numpy())
            #     print(part_label[i])
            #     print(part_score[i])
            #     plt.show()
            cur_score = final_mask = torch.zeros([seg_masks.shape[-2] , seg_masks.shape[-1]], dtype=torch.float, device=seg_masks.device)
            final_mask = torch.zeros([seg_masks.shape[-2] , seg_masks.shape[-1]], dtype=torch.uint8, device=seg_masks.device)
            for j in range(keep.sum()):
                final_mask[part_mask[j]]=int(part_label[j])+1
                
                cur_score[part_mask[j]]=part_score[j]
            
            final_mask = final_mask * human_seg_masks[i]
            cur_score = cur_score * human_seg_masks[i]
            cur_score = torch.sum(cur_score)/(cur_score!=0).sum()
            cur_score = cur_score * human_cate_scores [i]
            for j in range(self.num_classes):
                img = final_mask==(j+1)                
                #img=self.largestConnectComponent(img.cpu().numpy())
                if img.sum()<cfg.mix_pixels:
                    final_mask[final_mask==(j+1)]=0              
            if final_mask.sum()==0:
                continue
            
            
            
            for j in torch.unique(final_mask):
                if j ==0:
                    continue
                else:
                    instance_seg_masks.append((final_mask==j).unsqueeze(0))
                    instance_cate_labels.append(int(j-1))
                    instance_cate_scores.append(cur_score)
            # plt.imshow(final_mask.cpu().numpy())
            # print('after')
            # plt.show()
            ys, xs = np.where(final_mask.cpu().numpy() > 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            det=(x1, y1, x2, y2, cur_score.cpu().numpy())
            final_mask = scipy.sparse.csr_matrix(final_mask.cpu().numpy())
            mask_list.append(final_mask)
            det_list.append(det)

        if len(instance_seg_masks)==0:
            return (img_name,None_results) , None

        instance_seg_masks = torch.cat(instance_seg_masks)
        instance_cate_labels = torch.Tensor(instance_cate_labels)
        instance_cate_scores = torch.Tensor(instance_cate_scores)

        results['MASKS']=mask_list
        results['DETS']=det_list
        results['INSTANCE']=(instance_seg_masks,instance_cate_labels,instance_cate_scores)

        
        return (img_name,results) , None






