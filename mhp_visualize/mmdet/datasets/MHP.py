# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
from mmcv.runner import get_dist_info
import glob
import os
import os.path as osp
import tempfile
import torch
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.utils import print_log
from ..core.evaluation import iou_eval, SegmentationMetric, get_data
from .builder import DATASETS
from .coco import CocoDataset
import time
import pdb
import warnings
from mmdet.core.evaluation import eval_seg_ap
from numpy import *
@DATASETS.register_module()
class MHPDataset(CocoDataset):
    CLASSES = (
                'Cap/hat',
                'Helmet',
                 'Face',
                 'Hair',
                 'Left-arm',
                'Right-arm',
                 'Left-hand',
                 'Right-hand',
                 'Protector',
                 'Bikini/bra',
                 'Jacket/windbreaker/hoodie ',
                 'Tee-shirt',
                 'Polo-shirt',
                 'Sweater',
                 'Singlet',
                 'Torso-skin',
                 'Pants',
                 'Shorts/swim-shorts',
                 'Skirt',
                 'Stockings',
                 'Socks',
                 'Left-boot',
                 'Right-boot',
                 'Left-shoe',
                 'Right-shoe',
                 'Left-highheel',
                 'Right-highheel',
                 'Left-sandal',
                 'Right-sandal',
                 'Left-leg',
                 'Right-leg',
                 'Left-foot',
                 'Right-foot',
                 'Coat',
                 'Dress',
                 'Robe',
                 'Jumpsuit',
                 'Other-full-body-clothes',
                 'Headwear',
                 'Backpack',
                 'Ball',
                 'Bats',
                 'Belt',
                 'Bottle',
                 'Carrybag',
                 'Cases',
                 'Sunglasses',
                 'Eyewear',
                 'Glove',
                 'Scarf',
                 'Umbrella',
                 'Wallet/purse',
                 'Watch',
                 'Wristband',
                 'Tie',
                 'Other-accessary',
                 'Other-upper-body-clothes',
                 'Other-lower-body-clothes')

    def _filter_imgs(self, min_size=32):  # √
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):  # √
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
                bboxes_ignore, labels, masks, seg_map. \
                "masks" are already decoded into binary masks.
        """
        gt_instances = []
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                print(ann['category_id'], "not in ", self.cat_ids)
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                gt_instances.append(ann['instance_id'])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            masks=gt_masks_ann,
            instances=gt_instances,
            seg_map=img_info['segm_file'])

        return ann

    def results2txt(self, results, outfile_prefix):  # original cityscapes,did not change
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            list[str]: Result txt files which contains corresponding \
                instance segmentation images.
        """
        try:
            import cityscapesscripts.helpers.labels as CSLabels
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        result_files = []
        os.makedirs(outfile_prefix, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.data_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            # segm results
            if isinstance(segm_result, tuple):
                # Some detectors use different scores for bbox and mask,
                # like Mask Scoring R-CNN. Score of segm will be used instead
                # of bbox score.
                segms = mmcv.concat_list(segm_result[0])
                mask_score = segm_result[1]
            else:
                # use bbox score for mask score
                segms = mmcv.concat_list(segm_result)
                mask_score = [bbox[-1] for bbox in bboxes]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            prog_bar.update()
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = mask_score[i]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(outfile_prefix,
                                            basename + f'_{i}_{classes}.png')
                    mmcv.imwrite(mask, png_filename)
                    fout.write(f'{osp.basename(png_filename)} {class_id} '
                               f'{score}\n')
            result_files.append(pred_txt)

        return result_files

    def format_results(self, results, txtfile_prefix=None):  # original cityscapes,didn't change
        """Format the results to txt (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving txt/png files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2txt(results, txtfile_prefix)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='segm',
                 logger=None,
                 outfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None):  # need to channge
        """Evaluation in Cityscapes/COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None): The prefix of output file. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with COCO protocol, it would be the
                prefix of output json file. For example, the metric is 'bbox'
                and 'segm', then json files would be "a/b/prefix.bbox.json" and
                "a/b/prefix.segm.json".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output txt/png files. The output files would be
                png images under folder "a/b/prefix/xxx/" and the file name of
                images would be written into a txt file
                "a/b/prefix/xxx_pred.txt", where "xxx" is the video name of
                cityscapes. If not specified, a temp file will be created.
                Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric or cityscapes mAP \
                and AP@50.
        """
        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]

        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, outfile_prefix, logger))
            metrics.remove('cityscapes')

        # left metrics are all coco metric
        data_root = './data/MHP/'
        PREDICT_DIR = './apr'
        INST_PART_GT_DIR = './data/MHP/val/Instance_part_val'
        IOU_THRE = [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4 ]
        #IOU_THRE = [0.5]
        APR_CLASSES = ('background', 
                'Cap/hat',
                'Helmet',
                 'Face',
                 'Hair',
                 'Left-arm',
                'Right-arm',
                 'Left-hand',
                 'Right-hand',
                 'Protector',
                 'Bikini/bra',
                 'Jacket/windbreaker/hoodie ',
                 'Tee-shirt',
                 'Polo-shirt',
                 'Sweater',
                 'Singlet',
                 'Torso-skin',
                 'Pants',
                 'Shorts/swim-shorts',
                 'Skirt',
                 'Stockings',
                 'Socks',
                 'Left-boot',
                 'Right-boot',
                 'Left-shoe',
                 'Right-shoe',
                 'Left-highheel',
                 'Right-highheel',
                 'Left-sandal',
                 'Right-sandal',
                 'Left-leg',
                 'Right-leg',
                 'Left-foot',
                 'Right-foot',
                 'Coat',
                 'Dress',
                 'Robe',
                 'Jumpsuit',
                 'Other-full-body-clothes',
                 'Headwear',
                 'Backpack',
                 'Ball',
                 'Bats',
                 'Belt',
                 'Bottle',
                 'Carrybag',
                 'Cases',
                 'Sunglasses',
                 'Eyewear',
                 'Glove',
                 'Scarf',
                 'Umbrella',
                 'Wallet/purse',
                 'Watch',
                 'Wristband',
                 'Tie',
                 'Other-accessary',
                 'Other-upper-body-clothes',
                 'Other-lower-body-clothes')
        dat_list = get_data(data_root, 'val')
        
        app_results={}
        for pred in results:        
            app_results[pred[0]] = pred[1]
        apr_results={}
        for pred in results:        
            apr_results[pred[0]] = pred[2]

        threshold=[0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4 ]
        app_list=[]
        pcp_list=[]
        if len(metrics) > 0:
            for threshold_single in threshold:
                parsing_result = eval_seg_ap(app_results, dat_list, nb_class=len(self.CLASSES)+1, ovthresh_seg=threshold_single, From_pkl=True, Sparse=True)
                eval_results.update(parsing_result)
                app_list.append(parsing_result['AP_seg'])
                pcp_list.append(parsing_result['PCP'])
            print('app_list:',app_list)
            print('pcp_list:',pcp_list)
            app_vol = mean(app_list)
            pcp_vol = mean(pcp_list)
            print('app_vol, pcp_vol',app_vol,pcp_vol)
            # image_id_list = [x[:-4] for x in os.listdir(PREDICT_DIR) if x[-3:] == 'txt']
            # APr = np.zeros((len(APR_CLASSES) - 1, len(IOU_THRE)))

            # for ind in range(1, len(APR_CLASSES)):
            #     APr[ind - 1, :] = compute_class_ap(apr_results, image_id_list, ind, IOU_THRE,INST_PART_GT_DIR,PREDICT_DIR, APR_CLASSES)
            # print("-----------------AP-----------------")
            # print(APr)
            # print("-------------------------------------")
            # mAP = np.nanmean(APr, axis=0)
            # print("-----------------mAP-----------------")
            # print(mAP)
            
            # APrvol = np.nanmean(mAP)
            # print(np.nanmean(APrvol))
            # print("-------------------------------------")
            # eval_results.update({'APR':mAP, 'APrvol':APrvol})

        return eval_results


    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """

        k = (gt_label >= 0) & (gt_label < class_num)
        confusion_matrix = np.bincount(class_num * gt_label[k].astype(int) + pred_label[k],
                                       minlength=class_num ** 2).reshape(class_num, class_num)
        '''index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))
        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):

                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]'''  # wrong

        return confusion_matrix

    def evaluatesem(self, result_sem, infos, logger=None, num_class=None):
        msg = f'Evaluating semantic...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)
        result = {}
        assert len(result_sem) == len(infos)
        confusion_matrix = np.zeros((num_class + 1, num_class + 1))
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(infos))
        time.sleep(2)
        for i in range(len(infos)):
            info = infos[i]
            label = np.ones(info['masks'][0]['size'], dtype=np.uint8) * num_class
            for j in range(len(info['masks'])):
                mask_single = maskUtils.decode(info['masks'][j])
                label_single = info['labels'][j]
                label[np.nonzero(mask_single)[0], np.nonzero(mask_single)[1]] = label_single

            pred = np.ones_like(label) * num_class
            pred_encoder = result_sem[i]
            for j in range(len(pred_encoder)):
                mask_single = maskUtils.decode(pred_encoder[j])
                pred[np.nonzero(mask_single)[0], np.nonzero(mask_single)[1]] = j
            label = label.flatten()
            pred = pred.flatten()
            confusion_matrix += self.get_confusion_matrix(label, pred, num_class + 1)
            if rank == 0:
                prog_bar.update()
            # acc, mIoU=iou_eval(pred,label,num_class=num_class+1)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pixel_accuracy = tp.sum() / (pos.sum() + 0.00001)
            mean_accuracy = np.nanmean(tp / pos)
            IU_array = (tp / (pos + res - tp + 0.000001))
            mIoUs = np.nanmean(IU_array)
        print_log('mIoUs:{},pixel_accuracy:{},mean_accuracy:{}'.format(mIoUs, pixel_accuracy, mean_accuracy))
        classname = list(self.CLASSES)
        classname.append('background')
        for i, IU in enumerate(IU_array):
            print_log('{}:{}'.format(classname[i], IU))
        # result['IoUClass']=IU_array
        result['mIoUs'] = mIoUs
        result['pixel_accuracy'] = pixel_accuracy
        result['mean_accuracy'] = mean_accuracy
        return result

    def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of output txt file
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: Cityscapes evaluation results, contains 'mAP' \
                and 'AP@50'.
        """

        try:
            import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, txtfile_prefix)

        if tmp_dir is None:
            result_dir = osp.join(txtfile_prefix, 'results')
        else:
            result_dir = osp.join(tmp_dir.name, 'results')

        eval_results = {}
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
        CSEval.args.predictionPath = os.path.abspath(result_dir)
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = os.path.join(result_dir,
                                                   'gtInstances.json')
        CSEval.args.groundTruthSearch = os.path.join(
            self.img_prefix.replace('leftImg8bit', 'gtFine'),
            '*/*_gtFine_instanceIds.png')

        groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
        assert len(groundTruthImgList), 'Cannot find ground truth images' \
                                        f' in {CSEval.args.groundTruthSearch}.'
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
