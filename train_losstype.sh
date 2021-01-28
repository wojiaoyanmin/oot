#!/usr/bin/env bash

PORT=${PORT:-29500}

cd add_sharefeature_20_BCE
# sleep 60s
echo "add_sharefeature_20_BCE"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --no-validate --launcher pytorch
sleep 60s

echo "test"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_1x_MHP.py work_dirs/solo_r50_fpn_1x_MHP/latest.pth --eval segm --launcher pytorch

cd ..
sleep 60s
cd add_sharefeature_20_FL
echo "add_sharefeature_20_FL"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_MHP.py --no-validate --launcher pytorch
sleep 60s
echo "test"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/test.py configs/solo/solo_r50_fpn_1x_MHP.py work_dirs/solo_r50_fpn_1x_MHP/latest.pth --eval segm --launcher pytorch

echo "done"