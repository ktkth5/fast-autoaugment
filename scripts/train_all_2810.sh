python3 train.py --model=wideresnet28_10 --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=reduced_cifar_repro --log=28_fa
python3 train.py --model=wideresnet28_10 --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=default --log=28_baseline
python3 train.py --model=wideresnet28_10 --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=default --log=28_cutout --cutout=16
python3 train.py --model=wideresnet28_10 --epochs=200 --batch_size=128 --gpu=1 --lr=0.1 --aug=autoaug_paper_cifar10 --log=28_aa