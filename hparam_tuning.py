import os
import subprocess
from itertools import product

import yaml

if __name__ == "__main__":
    hparam_dict = None  # type: dict | None
    with open("hparams.yaml") as fp:
        try:
            hparam_dict = yaml.safe_load(fp)
        except yaml.YAMLError as e:
            print(e)
            exit(1)

    config_path = hparam_dict["CONFIG_PATH"]
    dataset = hparam_dict["DATASET"]
    del hparam_dict["CONFIG_PATH"]
    del hparam_dict["DATASET"]

    options_list = []  # type: list[list[tuple[str, str]]]

    for opt_key, opt_val_list in hparam_dict.items():
        options_list.append([(opt_key, v) for v in opt_val_list])
    for setting in product(*options_list):
        options_str_list = [f"{k}={v}" for k, v in setting]
        experiment_name = "-".join(f"{k.rsplit('.')[-1].lower()}_{v}" for k, v in setting)
        subprocess.run(["python", "train.py",
                        "--config_path", config_path,
                        "--name", experiment_name,
                        "--options", *options_str_list])
        subprocess.run(["python", "eval.py",
                        "--experiment_dir", os.path.join("logs", f"{dataset}_runs", experiment_name)])
