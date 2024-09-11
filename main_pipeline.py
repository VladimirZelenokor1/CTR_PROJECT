import logging
import sys
import json
import argparse
import pandas as pd
import datetime

from ctr_project.entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from ctr_project.data.make_dataset import read_data

from ctr_project.features.build_transformer import build_transformer, build_ctr_transformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def train_pipeline(config_path: str):
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(config_path)

    data: pd.DataFrame = read_data(training_pipeline_params.input_data_path)
    data['hour'] = data.hour.apply(lambda val: datetime.datetime.strptime(str(val), "%y%m%d%H"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)