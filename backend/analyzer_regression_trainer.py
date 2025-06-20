import pandas as pd
import numpy as np
from typing import List
from backend.feature_generator import FeatureGenerator
from backend.label_generator import LabelGenerator
from backend.regression_model_trainer import RegressionModelTrainer

class AnalyzerRegressionTrainer:
    def __init__(self, raw_df: pd.DataFrame, feature_group: List[str]):
        self.raw_df = raw_df.copy()
        self.feature_group = feature_group

    def train_and_predict(self) -> pd.DataFrame:
        # 1. Generate all features
        feature_generator = FeatureGenerator()
        df_features = feature_generator.create_all_features(self.raw_df.copy())
        # 2. Generate Next_High label
        label_generator = LabelGenerator()
        df_labeled = label_generator.elasticnet_label_next_high(df_features)
        # 3. Drop NaNs, keep only needed columns
        keep_cols = self.feature_group + ['Next_High', 'Close', 'Date', 'Time']
        df_labeled = df_labeled[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
        # 4. Train/test split: last 100 rows as test
        train_df = df_labeled.iloc[:-100]
        test_df = df_labeled.iloc[-100:]
        X_train = train_df[self.feature_group]
        y_train = train_df['Next_High']
        X_test = test_df[self.feature_group]
        # 5. Train ElasticNet and predict
        reg = RegressionModelTrainer(include_prices=False, apply_filter=False)
        reg.x_train = X_train
        reg.y_train = y_train
        reg.x_test = X_test
        reg.y_test = test_df['Next_High']
        reg.train_model()
        reg.make_predictions()
        # 6. Insert predictions into a copy of the original df
        result_df = self.raw_df.copy()
        # Align predictions to the last 100 rows (by index)
        pred_indices = test_df.index
        result_df['predicted_high'] = np.nan
        result_df.loc[pred_indices, 'predicted_high'] = reg.predictions
        return result_df
