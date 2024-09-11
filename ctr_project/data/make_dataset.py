from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from ctr_project.entities.split_params import SplittingParams

def read_data(input_data_path: str) -> pd.DataFrame:
    return pd.read_csv(input_data_path)

def split_data(data: pd.DataFrame, split_params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=split_params.val_size, random_state=split_params.random_state
    )

    return train_data, val_data

