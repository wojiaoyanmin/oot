#!/usr/bin/env bash

PORT=${PORT:-29500}

cd sharefeature_20
echo "sharefeature_20 dcn_1x"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_dcn_MHP.py  --resume-from work_dirs/solo_r50_fpn_1x_dcn_MHP/latest.pth --no-validate --launcher pytorch


sleep 60s
echo "sharefeature_20 dcn_3x"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
     ./tools/train.py configs/solo/solo_r50_fpn_3x_dcn_MHP.py --resume-from work_dirs/solo_r50_fpn_3x_dcn_MHP/epoch_9.pth --no-validate --launcher pytorch

