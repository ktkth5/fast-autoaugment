# done on aws p3.xlarge
# rm -r ~/ray_results/TrainCIFAR/
# python3 main.py --B=200 --T=2 --num_samples=40 --max_concurrent=100 --epochs=100 --batch_size=128 --cpu=1 --gpu=0.1 \
#                 --split_idx=0 --my_home --pass_train
# python3 main.py --B=200 --T=2 --num_samples=40 --max_concurrent=100 --epochs=100 --batch_size=128 --cpu=1 --gpu=0.1 \
#                 --split_idx=1 --my_home
# python3 main.py --B=200 --T=2 --num_samples=40 --max_concurrent=100 --epochs=100 --batch_size=128 --cpu=1 --gpu=0.1 \
#                 --split_idx=2 --my_home
# python3 main.py --B=200 --T=2 --num_samples=40 --max_concurrent=100 --epochs=100 --batch_size=128 --cpu=1 --gpu=0.1 \
#                 --split_idx=3 --my_home
python3 main.py --B=200 --T=2 --num_samples=40 --max_concurrent=100 --epochs=100 --batch_size=128 --cpu=1 --gpu=0.1 \
                --split_idx=4 --my_home
# python3 bestparams.py --exp_dir=/home/kento/ray_results/TrainCIFAR --K=10 --log_dir=checkpoint