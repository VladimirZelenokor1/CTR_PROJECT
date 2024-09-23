import yaml
import logging
import sys
import os
from pathlib import Path

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from ctr_project.entities.train_params import TrainingParams
from ctr_project.entities.split_params import SplittingParams
from ctr_project.entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

BASE_DIR = Path(__file__).resolve().parent.parent
PATH = os.path.join(os.getcwd(), 'configs/train.yaml')

@dataclass()
class TrainingPipelineParams:
    output_model_path: str
    output_transformer_path: str
    output_ctr_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_path: str
    input_preprocessed_data_path: str
    use_mlflow: bool = field(default=True)


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)

        config_dict['output_model_path'] = str((BASE_DIR / config_dict['output_model_path']).resolve()).replace('\\', '/')
        config_dict['output_transformer_path'] = str((BASE_DIR / config_dict['output_transformer_path']).resolve()).replace('\\', '/')
        config_dict['output_ctr_transformer_path'] = str((BASE_DIR / config_dict['output_ctr_transformer_path']).resolve()).replace('\\', '/')
        config_dict['metric_path'] = str((BASE_DIR / config_dict['metric_path']).resolve()).replace('\\', '/')
        config_dict['input_data_path'] = str((BASE_DIR / config_dict['input_data_path']).resolve()).replace('\\', '/')
        config_dict['input_preprocessed_data_path'] = str((BASE_DIR / config_dict['input_preprocessed_data_path']).resolve()).replace('\\', '/')

        schema = TrainingPipelineParamsSchema().load(config_dict)
        return schema


if __name__ == "__main__":
    params = read_training_pipeline_params(PATH)