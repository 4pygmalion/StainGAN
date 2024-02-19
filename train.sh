"""
아래와 같은 이미지 구성으로 있어야함
/home/heon/repositories/StainGAN/data/images
├── 3dh
│   ├── test_3dh -> /home/heon/heon_vast/inhouse_tiling_3dh/test/all
│   └── train_3dh -> /home/heon/heon_vast/inhouse_tiling/train/all
└── leica
    ├── test_leica -> /home/heon/heon_vast/inhouse_tiling/test/all
    └── train_leica -> /home/heon/heon_vast/inhouse_tiling/train/all

"""


python3 train.py \
    --dataroot data/images \
    --gpu_ids 1 \
    --loadSize 512 \
    --phaseA 3dh \
    --phaseB leica \
    --batchSize 8 \
    --name 3dhtoleica