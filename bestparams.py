import os
import glob
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir",default="/home/ubuntu/ray_results/TrainCIFAR",
                    help="path to the experiment directory")
parser.add_argument("--K", default=3, type=int)
parser.add_argument("--log_dir", default="checkpoint")

args = parser.parse_args()

assert os.path.exists(args.exp_dir)


for split_idx in range(5):

    _log_dirs = glob.glob(f"{args.exp_dir}/**/result.json")
    log_dirs = []
    for path in _log_dirs:
        try:
            js = json.load(open(path, "rb"))
        except:
            continue
        if f"split_idx={split_idx}" in js["config"]["args"]:
            log_dirs.append(path)
    print(f"split idx {split_idx}\tlog num: {len(log_dirs)}")
    augs = []
    augs_acc = []

    for log in log_dirs:
        js = json.load(open(log, "rb"))
        aug1, aug2 = js["config"]["aug1"], js["config"]["aug2"]
        acc = js["mean_accuracy"]
        p1, v1 = js["config"]["p1"],  js["config"]["value1"]
        p2, v2 =  js["config"]["p2"],  js["config"]["value2"]
        augs.append([aug1, aug2, acc, p1, v1, p2, v2])

    unique_augs = []

    for aug in augs:
        aug = aug[:2]
        if aug not in unique_augs:
            unique_augs.append(aug)
            # print(aug)


    best_params_each_aug = []

    for aug in unique_augs:
        best_acc = 0
        for result in augs:
            if not aug == result[:2]:
                continue
            if best_acc < result[2]:
                best_param = result
        best_params_each_aug.append(best_param)


    # Print Top K
    best_param = sorted(best_params_each_aug, key=lambda x:float(x[2]), reverse=True)
    log_path = os.path.join(args.log_dir, f"{args.exp_dir.split('/')[-2]}.txt")

    with open(log_path, "a") as f:
        for i, param in enumerate(best_param):
            if i == args.K:
                break
            f.write(",".join(list(map(str,param))))
            f.write("\n")



