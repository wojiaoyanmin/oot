#!/usr/bin/env bash

PORT=${PORT:-29500}

cd sharefeaturekernel_20
echo "sharefeaturekernel_20"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py  --resume-from work_dirs/solo_r50_fpn_1x_MHP/latest.pth --no-validate --launcher pytorch

cd ..
cd sharefeature_20
sleep 60s
echo "sharefeature_20"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
     ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --no-validate --launcher pytorch

