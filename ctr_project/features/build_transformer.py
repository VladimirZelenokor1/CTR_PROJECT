import sys
import pandas as pd
import logging
import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from ctr_project.entities.feature_params import FeatureParams
from ctr_project.features.DeviceCountTransformer import DeviceCountTransformer
from ctr_project.features.CtrTransformer import CtrTransformer
from ctr_project.features.UserCountTransformer import UserCountTransformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def build_transformer() -> Pipeline:

    time_transformer = FunctionTransformer(lambda df: pd.DataFrame(
        {
            "hour_of_day": df["hour"].dt.hour,
            "day_of_week": df["hour"].dt.dayofweek,
            "device_ip": df["device_ip"],
            "device_id": df["device_id"],
        }
    ))

    original_features = ["id", "hour", "device_ip", "device_id"]
    device_time_transformer = ColumnTransformer(
        transformers=[
            (
                "device_ip_count",
                DeviceCountTransformer("device_ip"),
                original_features,
            ),
            (
                "device_id_count",
                DeviceCountTransformer("device_id"),
                original_features,
            ),
            ("time_transformer", time_transformer, original_features[1:]),
        ],
    )

    user_count_transformer = UserCountTransformer()

    pipeline = Pipeline(
        [
            ("device_time_transformer", device_time_transformer),
            ("user_count_transformer", user_count_transformer),
        ]
    )
    logger.info(f"Transformer pipeline built successfully. {pipeline}")

    return pipeline

def build_ctr_transformer(params: FeatureParams) -> CtrTransformer:

    feature_names = params.ctr_features
    ctr_transformer = CtrTransformer(features=feature_names)
    logger.info(f"CTR transformer built successfully. {ctr_transformer}")

    return ctr_transformer

def process_count_features(
        transformer: Pipeline, df: pd.DataFrame, params: FeatureParams = None
) -> pd.DataFrame:

    counts_df = transformer.fit_transform(df)
    return pd.concat([df, counts_df[params.count_features]], axis=1)

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
