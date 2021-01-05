#!/usr/bin/env bash

PORT=${PORT:-29500}

cd share_feature_20_pascal
echo "share_feature_20_pascal50"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_dcn_PAS.py  --resume-from work_dirs/solo_r50_fpn_1x_dcn_PAS/latest.pth --no-validate --launcher pytorch


sleep 60s
echo "share_feature_20_pascal50-validate"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
     ./tools/test.py configs/solo/solo_r50_fpn_1x_dcn_PAS.py work_dirs/solo_r50_fpn_1x_dcn_PAS/latest.pth --eval segm --launcher pytorch

sleep 60s
echo "share_feature_20_pascal101"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r101_fpn_1x_dcn_PAS.py --resume-from work_dirs/solo_r101_fpn_1x_dcn_PAS/latest.pth --no-validate --launcher pytorch

sleep 60s
echo "share_feature_20_pascal101-validate"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
     ./tools/test.py configs/solo/solo_r101_fpn_1x_dcn_PAS.py work_dirs/solo_r101_fpn_1x_dcn_PAS/latest.pth --eval segm --launcher pytorch

