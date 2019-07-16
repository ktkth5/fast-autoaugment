# fast-autoaugment

This is PyTorch Implementation of Fast AutoAugment()

### Installation
1. clone this repository
2. install ray-project (https://ray.readthedocs.io/en/latest/installation.html)
3. pip install -r requirements.txt
4. comment out /ray/tune/trial_runner.py:293.py
    ```
    def checkpoint(self):
        ...
        """
        runner_state = {
            "checkpoints": list(
                self.trial_executor.get_checkpoints().values()),
            "runner_data": self.__getstate__(),
            "stats": {
                "start_time": self._start_time,
                "timestamp": time.time()
            }
        }
        tmp_file_name = os.path.join(metadata_checkpoint_dir,
                                     ".tmp_checkpoint")
        with open(tmp_file_name, "w") as f:
            json.dump(runner_state, f, indent=2, cls=_TuneFunctionEncoder)

        os.rename(
            tmp_file_name,
            os.path.join(metadata_checkpoint_dir,
                         TrialRunner.CKPT_FILE_TMPL.format(self._session_str)))
        """
        return metadata_checkpoint_dir
    ```
5. run script on wideresnet40-2  
    to search policy:
    ```shell
    scripts/full_search.sh
    ```  
    then, find best parameters of data augment:
    ```shell
    python3 bestparams.py --exp=[path to ray experiment folder] \
    --K=[Top k policies of each splits] --log_dir=[path to log folder]
    ```
    finaly, train full cifar10 with searched data augment policy
    ```shell
    scripts/train_cifar10.sh
    ```
6. you can also re-product score on cifa10 with Baseline/Cutout/AutoAugment
    ```shell
    script/train_cifar10_baseline.sh # baseline
    script/train_cifar10_cutout.sh # cutout
    script/train_cifar10_autoaug.sh # autoaug
    ```