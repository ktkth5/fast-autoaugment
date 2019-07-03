# done on mac
python3 main.py --B=5 --T=2 --num_samples=8 --max_concurrent=10 --epochs=5 --batch_size=8 --cpu=4 --gpu=0 --verbose=0
python3 bestparams.py --exp_dir=/Users/kento/ray_results/TrainCIFAR --K=10 --log_dir=checkpoint