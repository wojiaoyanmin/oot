#!/usr/bin/env bash

PORT=${PORT:-29500}

cd share_feature_20_pascal
# echo "share_feature_20_pascal50-validate"
# for i in 30 32 34 36 38 42 44 45 46 47 48 49 52 51 52 53 54 55 56 57 58 59 60:
# do
# echo $i
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
#      ./tools/test.py configs/solo/solo_r50_fpn_1x_dcn_PAS.py work_dirs/solo_r50_fpn_1x_dcn_PAS/epoch_$i.pth --eval segm --launcher pytorch
# done
# echo "share_feature_20_pascal50-train"
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
#     ./tools/train.py configs/solo/solo_r50_fpn_1x_dcn_PAS.py --resume-from work_dirs/solo_r50_fpn_1x_dcn_PAS/latest.pth --launcher pytorch


# sleep 60s
echo "share_feature_20_pretrain"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_dcn_prePAS.py --launcher pytorch

echo "share_feature_20_3X_TEST"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_3x_dcn_MHP.py work_dirs/solo_r50_fpn_3x_dcn_MHP/epoch_36.pth --eval segm --launcher pytorch
sleep 60s
echo "share_feature_20_pascal101"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r101_fpn_1x_dcn_PAS.py --resume-from work_dirs/solo_r101_fpn_1x_dcn_PAS/latest.pth --no-validate --launcher pytorch
sleep 60s
echo "share_feature_20_pas_TEST"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r101_fpn_1x_dcn_PAS.py work_dirs/solo_r101_fpn_1x_dcn_PAS/latest.pth --eval segm --launcher pytorch