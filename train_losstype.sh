#!/usr/bin/env bash

PORT=${PORT:-29500}

cd share_feature_20_pascal_54epoch
# sleep 60s
echo "share_feature_20_pascal_54epoch"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    ./tools/train.py configs/solo/solo_r50_fpn_1x_PAS.py --launcher pytorch
cd ..
sh CIHP.sh


echo "done"