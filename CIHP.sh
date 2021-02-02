#!/usr/bin/env bash

PORT=${PORT:-29500}


cd sharefeature_20
echo "sharefeature_20"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_3x_dcn_CIHP.py --resume-from work_dirs/solo_r50_fpn_3x_dcn_CIHP/latest.pth --no-validate --launcher pytorch

echo "test"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_3x_dcn_CIHP.py work_dirs/solo_r50_fpn_3x_dcn_CIHP/latest.pth --eval segm --launcher pytorch
echo "done"