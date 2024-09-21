import logging
import sys
from time import sleep
import numpy as np
import requests

from ctr_project.data.make_dataset import read_data
from ctr_project.entities.train_pipeline_params import  (
    read_training_pipeline_params,
    TrainingPipelineParams,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

if __name__ == '__main__':
    config_path = "D:/Projects/CTR_PROJECT/configs/train.yaml"
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )

    data = read_data(training_pipeline_params.input_preprocessed_data_path)

    for i in range(9):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]

        response = requests.post(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": [list(data.columns)]}
        )

        logger.info(response.status_code)

        sleep(1)


