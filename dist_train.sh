#!/usr/bin/env bash

PORT=${PORT:-29500}

cd shareall_20_pascal
echo "shareall_20_pascal train"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r101_fpn_1x_PAS.py  --resume-from work_dirs/solo_r101_fpn_1x_PAS/latest.pth --resume-from work_dirs/solo_r101_fpn_1x_PAS/latest.pth --no-validate --launcher pytorch


sleep 60s
echo "shareall_20_pascal test"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
     ./tools/test.py configs/solo/solo_r101_fpn_1x_PAS.py work_dirs/solo_r101_fpn_1x_PAS/latest.pth --eval segm --launcher pytorch

