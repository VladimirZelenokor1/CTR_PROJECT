import sys
import logging
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class UserCountTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to count the number of unique device_ips for each hour of the day.
    """

    def __init__(self):
        self.user_count_feature = []

    def fit(self, X: pd.DataFrame, y=None):
        # Assuming 'hour_of_day', 'device_ip', and 'device_id' are the columns in X
        hour_of_day = X[:, 2]
        device_ip = X[:, 4]

        data_group = {}

        # Count the number of unique device_ips for each hour of the day
        for hour in np.unique(hour_of_day):
            data_group[hour] = (device_ip[hour_of_day == hour]).shape[0]

        for hour in hour_of_day:
            self.user_count_feature.append(data_group[hour])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Assuming 'device_ip', 'device_id', and 'hour_of_day' are the columns in X
        result_array = np.array(self.user_count_feature).reshape(-1, 1)
        count_df = pd.DataFrame(result_array, columns=[
            'device_ip_count',
            'device_id_count',
            'hour_of_day',
            'day_of_week',
            'device_ip',
            'device_id',
        ])

        count_df['hourly_user_count'] = result_array

        return count_df