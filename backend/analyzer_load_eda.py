import pickle
import pandas as pd

class ModelLoaderAndExplorer:
    def __init__(self, regression_path, classifier_path):
        self.regression_path = regression_path
        self.classifier_path = classifier_path

    def load_and_explore(self):
        # === Load Regression Model ===
        with open(self.regression_path, "rb") as f:
            regression_model = pickle.load(f)
        print("\n📦 Regression Model Attributes:")
        for attr in dir(regression_model):
            if not attr.startswith("__"):
                print(f"🔹 {attr}: {type(getattr(regression_model, attr))}")

        # === Load Classifier Model ===
        with open(self.classifier_path, "rb") as f:
            classifier_model = pickle.load(f)
        print("\n📦 Classifier Model Attributes:")
        for attr in dir(classifier_model):
            if not attr.startswith("__"):
                print(f"🔸 {attr}: {type(getattr(classifier_model, attr))}")

        # === Extract Classifier Predictions ===
        df_classifier_preds = classifier_model.classifier_predictions_df.copy()

        return regression_model, classifier_model, df_classifier_preds
