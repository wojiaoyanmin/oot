#!/usr/bin/env bash

PORT=${PORT:-29500}


cd add_sharefeature_20_32
echo "add_sharefeature_20_32"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --resume-from work_dirs/solo_r50_fpn_1x_MHP/latest.pth --no-validate --launcher pytorch

echo "add_sharefeature_20_32 test"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_1x_MHP.py work_dirs/solo_r50_fpn_1x_MHP/latest.pth --eval segm --launcher pytorch

sleep 60s
cd ..
cd add_sharefeature_20_BCE
echo "add_sharefeature_20_BCE"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --no-validate --launcher pytorch

echo "add_sharefeature_20_BCE test"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_1x_MHP.py work_dirs/solo_r50_fpn_1x_MHP/latest.pth --eval segm --launcher pytorch

sleep 60s
cd ..
cd add_sharefeature_20_FL
echo "add_sharefeature_20_FL"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --no-validate --launcher pytorch

echo "add_sharefeature_20_FLtest"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_1x_MHP.py work_dirs/solo_r50_fpn_1x_MHP/latest.pth --eval segm --launcher pytorch
echo "done"