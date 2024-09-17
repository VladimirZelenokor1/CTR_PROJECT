import logging
import sys
import json
import argparse
import pandas as pd
import datetime

from ctr_project.entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from ctr_project.data.make_dataset import read_data, split_data

from ctr_project.features.build_transformer import build_transformer, build_ctr_transformer, process_count_features, \
    extract_target, process_count_features
from ctr_project.modeling.model_fit_predict import train_model, predict_model, evaluate_model, serialize_model
from ctr_project.modeling.repro_experiments import log_experiment_mlflow

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def train_pipeline(config_path: str):
    # Read training pipeline parameters from the input configuration file and preprocess them
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(config_path)

    # Read data from the input data path and preprocess it by converting hour to datetime format and extracting hour
    data: pd.DataFrame = read_data(training_pipeline_params.input_data_path)
    data['hour'] = data.hour.apply(lambda val: datetime.datetime.strptime(str(val), "%y%m%d%H"))

    # Build transformer using the feature parameters and fit it to the data
    transformer = build_transformer()
    processed_data = process_count_features(
        transformer, data, training_pipeline_params.feature_params
    )

    # Split the data into training and validation sets using the splitting parameters
    train_df, val_df = split_data(
        processed_data, training_pipeline_params.splitting_params
    )

    # Build CTR transformer using training data and fit it to the training data
    ctr_transformer = build_ctr_transformer(training_pipeline_params.feature_params)
    ctr_transformer.fit(train_df)

    # Transform training data using CTR transformer and extract target variable
    train_features = ctr_transformer.transform(train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    # Transform validation data using CTR transformer and evaluate model performance on validation data
    val_features = ctr_transformer.transform(val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    # Train a model using the training data and evaluate its performance on validation data
    if training_pipeline_params.use_mlflow:
        model, metrics = log_experiment_mlflow(
            run_name="first_run",
            train_features=train_features,
            train_target=train_target,
            val_features=val_features,
            val_target=val_target,
            training_pipeline_params=training_pipeline_params,
        )
    else:
        model = train_model(
            train_features, train_target, training_pipeline_params.train_params
        )

        predicted_proba, preds = predict_model(model, val_features)
        metrics = evaluate_model(predicted_proba, preds, val_target)
        logger.debug(f"preds/ targets shapes:  {(preds.shape, val_target.shape)}")

    # Save metrics to file
    with open(training_pipeline_params.metric_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)

    # Save trained model and CTR transformer to files
    serialize_model(model, training_pipeline_params.output_model_path)
    serialize_model(
        ctr_transformer, training_pipeline_params.output_ctr_transformer_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)