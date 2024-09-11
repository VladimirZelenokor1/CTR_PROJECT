import sys
import logging
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class CtrTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate CTR (Click-Through-Rate) for each feature in the dataset.
    """

    def __init__(self, features: list = None):
        self.ctr_df = None
        self.mean_ctr = dict()
        self.vocab = dict()
        self.features = features

    def _response_fit(self, data, feature_name):
        # Group by feature_name and click to calculate ctr
        df_vocab = data.groupby([feature_name, 'click']).size().unstack()
        df_vocab['ctr'] = df_vocab[1] / (df_vocab[0] + df_vocab[1])

        df_vocab.dropna(inplace=True)
        mean_ctr = df_vocab['ctr'].mean()

        # Store the ctr and vocab in a dictionary for quick lookup
        keys = df_vocab.index.tolist()
        values = df_vocab['ctr'].values.tolist()
        vocab = {keys[i]: values[i] for i in range(len(keys))}

        return vocab, mean_ctr

    def _response_transform(self, X: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        # Apply the CTR transformation for each feature in the given data
        vector = []
        for row in X:
            vector.append(self.vocab[feature_name].get(row, self.mean_ctr[feature_name]))

        return vector

    def fit(self, X:pd.DataFrame, y=None):
        # Fit the CTR model for each feature in the given data
        for column_name in self.features:
            vocab_feat, mean_ctr_feat = self._response_fit(X, column_name)
            self.vocab[column_name] = vocab_feat
            self.mean_ctr[column_name] = mean_ctr_feat

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply the CTR transformation to the given data and return a new DataFrame
        self.ctr_df = pd.DataFrame()

        for column_name in self.features:
            self.ctr_df[column_name] = self._response_transform(
                X[column_name],
                column_name
            )

        return self.ctr_df

