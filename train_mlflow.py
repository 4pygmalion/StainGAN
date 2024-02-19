"""
$ python3 train_mlflow.py \
    --artifact_uri file:///vast/AI_team/mlflow_artifact/4/6479f5480d784a939ed1863a8ba2740c/artifacts/patchset.pkl \
    --output_dir data/camelyon
"""

import os
import pickle
import argparse
import subprocess

import mlflow
from sklearn.model_selection import train_test_split
from core.data_model import Patches


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_uri", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
    return


TRACKING_URI = "http://219.252.39.224:5000"

if __name__ == "__main__":
    ARGS = get_args()
    mlflow.set_tracking_uri(TRACKING_URI)

    patcheset_path = mlflow.artifacts.download_artifacts(ARGS.artifact_uri)
    # patchset = load_pickle(patcheset_path)

    # a_domain_train, a_domain_test = train_test_split(
    #     patchset.train, test_size=0.2, random_state=42
    # )
    # os.makedirs("data/camelyon/rumc/train", exist_ok=True)
    # os.makedirs("data/camelyon/rumc/test", exist_ok=True)
    # Patches(a_domain_train).save("data/camelyon/rumc/train")
    # Patches(a_domain_test).save("data/camelyon/rumc/test")

    # b_domain_train, b_domain_test = train_test_split(
    #     patchset.test, test_size=0.2, random_state=42
    # )
    # os.makedirs("data/camelyon/umcu/train", exist_ok=True)
    # os.makedirs("data/camelyon/umcu/test", exist_ok=True)
    # Patches(b_domain_train).save("data/camelyon/umcu/train")
    # Patches(b_domain_test).save("data/camelyon/umcu/test")

    CMD = f"""
    python3 train.py \
        --dataroot {ARGS.output_dir} \    
        --gpu_ids 1 \
        --loadSize 512 \
        --phaseA umcu \
        --phaseB rumc \
        --batchSize 4 \
        --name umcu2rumc
    """
    subprocess.run(CMD, shell=True, check=True)
