python3 train.py --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=reduced_cifar_repro --log=fa
python3 train.py --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=default --log=baseline
python3 train.py --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=default --log=cutout --cutout=16