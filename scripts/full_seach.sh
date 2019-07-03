# done on aws p3.xlarge
python3 main.py --B=200 --T=2 --num_samples=200 --max_concurrent=1000 --epochs=100 --batch_size=64 --cpu=8 --gpu=1 --aws
python3 bestparams.py --exp_dir=/home/ubuntu/ray_results/TrainCIFAR --K=10 --log_dir=checkpoint