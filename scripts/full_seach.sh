# done on aws p3.xlarge
python3 main.py --B=50 --T=2 --num_samples=50 --max_concurrent=5 --epochs=100 --batch_size=64 --cpu=8 --gpu=1 --aws
python3 bestparams.py --exp_dir=/home/ubuntu/ray_results/TrainCIFAR --K=10 --log_dir=checkpoint