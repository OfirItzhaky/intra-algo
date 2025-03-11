from fastapi import FastAPI, Query
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
from fastapi.responses import Response
import matplotlib.pyplot as plt

from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor
from regression_model_trainer import RegressionModelTrainer
from label_generator import LabelGenerator
from feature_generator import FeatureGenerator
from data_loader import DataLoader

app = FastAPI()


# CORS for frontend
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DataLoader instance
data_loader = DataLoader()

# In-memory data holders
training_df_raw = None      # Original loaded data
training_df_features = None # Data after feature generation
training_df_labels = None   # Data after label generation
training_df_final = None    # Final processed data (used for training)
simulation_df = None
# In-memory data holders for regression
x_train_regression = None
y_train_regression = None
x_test_regression = None
y_test_regression = None
regression_trained_model = None
predictions_regression = None

@app.get("/load-data/")
def load_data(
    file_path: str = Query(...),
    data_type: str = Query(...),
    symbol: str = Query(...)
):
    global training_df_raw, training_df_features, training_df_labels, training_df_final, simulation_df

    base_folder = "data"
    full_path = os.path.join(base_folder, data_type, file_path)

    print(f"🔗 Loading file from: {full_path}")

    df = data_loader.load_from_csv(full_path)
    if df.empty:
        return {"status": "error", "message": f"{data_type.capitalize()} file could not be loaded or is empty."}

    if data_type == "training":
        training_df_raw = df  # ✅ Store raw data
        training_df_features = None
        training_df_labels = None
        training_df_final = None

    elif data_type == "simulating":
        if training_df_raw is None:
            return {"status": "error", "message": "Training data must be loaded before simulation data!"}

        try:
            df = data_loader.align_and_validate_simulation(training_df_raw, df)
            simulation_df = df  # ✅ Save aligned simulation data
        except ValueError as e:
            return {"status": "error", "message": str(e)}

    else:
        return {"status": "error", "message": "Invalid data type provided."}

    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    summary = {
        "symbol": symbol,
        "first_date": first_row["Date"],
        "first_time": first_row["Time"],
        "last_date": last_row["Date"],
        "last_time": last_row["Time"],
        "dataType": data_type  # Important for frontend to know
    }

    print(f"✅ Summary: {summary}")

    return {
        "status": "success",
        "summary": summary
    }

@app.get("/get-loaded-data/")
def get_loaded_data(data_type: str = Query(..., description="Data type - training or simulating")):
    if data_type == "training" and training_df_raw is not None:
        return {"status": "success", "data": training_df_raw.head(5).to_dict(orient="records")}
    elif data_type == "simulating" and simulation_df is not None:
        return {"status": "success", "data": simulation_df.head(5).to_dict(orient="records")}
    else:
        return {"status": "error", "message": f"No {data_type} data loaded yet."}

@app.post("/generate-features/")
def generate_features():
    global training_df_raw, training_df_features

    # Ensure training data exists before feature generation
    if training_df_raw is None:
        return {"status": "error", "message": "Training data is not loaded. Please load data first."}

    # Run feature generation
    feature_generator = FeatureGenerator()
    training_df_features = feature_generator.create_all_features(training_df_raw)  # ✅ Use `training_df_raw`

    # Count the number of new features
    original_cols = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
    num_new_features = len(set(training_df_features.columns) - original_cols)

    return {
        "status": "success",
        "message": "Feature generation completed successfully!",
        "new_features_count": num_new_features
    }

label_generator = LabelGenerator()

@app.get("/generate-labels/")
def generate_labels(label_type: str = Query(..., description="Label type (next_high or good_bar)")):
    global training_df_features, training_df_labels

    if training_df_features is None:
        return {"status": "error", "message": "Features must be generated first!"}

    if label_type == "next_high":
        training_df_labels = label_generator.elasticnet_label_next_high(training_df_features)  # ✅ Use `training_df_features`
    elif label_type == "good_bar":
        training_df_labels = label_generator.add_good_bar_label(training_df_features)  # ✅ Use `training_df_features`
    else:
        return {"status": "error", "message": "Invalid label type selected."}

    rows_labeled = len(training_df_labels)

    return {
        "status": "success",
        "summary": {
            "label_type": label_type,
            "rows_labeled": rows_labeled
        }
    }


@app.get("/get-regression-chart/")
def get_regression_chart():
    """
    Generates the regression chart dynamically and returns it as an image.
    """
    global trainer  # Ensure we're using the trainer instance

    if trainer is None or trainer.y_test is None or trainer.predictions is None:
        return {"status": "error", "message": "Visualization skipped due to missing data."}

    if not hasattr(trainer, "x_test_with_meta") or trainer.x_test_with_meta is None:
        return {"status": "error", "message": "Metadata (Date, Time, OHLC) is missing for visualization."}

    print("📊 Generating visualization for the last 1,000 bars...")

    processor = DataProcessor()
    fig = processor.visualize_regression_predictions_for_pycharm(
        trainer.x_test_with_meta, trainer.y_test, trainer.predictions, n=1000  # Default to 1,000 for scrolling
    )

    # ✅ Save to memory instead of disk
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format="png", facecolor="black")  # ✅ Dark background
    plt.close(fig)  # ✅ Prevent memory leaks
    img_bytes.seek(0)

    return Response(content=img_bytes.getvalue(), media_type="image/png")



# ✅ Define request model with `drop_price_columns`, `apply_filter`, and `filter_threshold`
class TrainRegressionRequest(BaseModel):
    drop_price_columns: bool = True  # Default to dropping prices
    apply_filter: bool = True       # Default: filtering only for metrics, not visuals.
    filter_threshold: float = 4.0     # Default threshold for filtering extreme errors

@app.post("/train-regression-model/")
def train_regression_model(request: TrainRegressionRequest):
    global training_df_labels, regression_trained_model, trainer  # Keep only necessary global vars

    if training_df_labels is None:
        return {"status": "error", "message": "Labeled training data is not available. Please generate labels first."}

    # ✅ Store Date, Time, Open, High, Low, Close before feature selection
    meta_columns = training_df_labels[["Date", "Time", "Open", "High", "Low", "Close"]].copy()

    # ✅ Initialize Trainer with user options
    trainer = RegressionModelTrainer(
        include_prices=not request.drop_price_columns,
        apply_filter=request.apply_filter,
        filter_threshold=request.filter_threshold
    )

    # ✅ Prepare Data
    trainer.prepare_data(training_df_labels)

    # ✅ Train Model
    trainer.train_model()
    regression_trained_model = trainer.model  # ✅ Save trained model globally

    # ✅ Make Predictions (populates `trainer.predictions`)
    trainer.make_predictions()

    # ✅ Ensure x_test_with_meta contains necessary columns
    trainer.x_test_with_meta = trainer.x_test.copy()  # Copy test data

    # ✅ Always merge Date & Time (avoid missing timestamps)
    trainer.x_test_with_meta = trainer.x_test_with_meta.join(meta_columns[["Date", "Time"]].loc[trainer.x_test.index])

    # ✅ Merge OHLC columns **only if dropping price columns**
    if request.drop_price_columns:
        trainer.x_test_with_meta = trainer.x_test_with_meta.join(meta_columns[["Open", "High", "Low", "Close"]].loc[trainer.x_test.index])

    # ✅ Compute prediction errors
    prediction_errors = abs(trainer.predictions - trainer.y_test)

    # ✅ Apply filter **only for metrics, NOT visualization**
    if request.apply_filter:
        valid_indices = prediction_errors <= request.filter_threshold
        trainer.y_test_filtered = trainer.y_test[valid_indices].copy()
        trainer.predictions_filtered = trainer.predictions[valid_indices].copy()
    else:
        # ✅ No filtering: Use all predictions for evaluation
        trainer.y_test_filtered = trainer.y_test.copy()
        trainer.predictions_filtered = trainer.predictions.copy()

    # ✅ Compute Metrics on Filtered Data
    mse_filtered = mean_squared_error(trainer.y_test_filtered, trainer.predictions_filtered)
    r2_filtered = r2_score(trainer.y_test_filtered, trainer.predictions_filtered)

    # ✅ Generate the visualization **directly here** using `x_test_with_meta`
    print("📊 Generating visualization for the last 20 bars...")
    processor = DataProcessor()
    processor.visualize_regression_predictions_for_pycharm(
        trainer.x_test_with_meta, trainer.y_test, trainer.predictions, n=20  # Show only last 20 by default
    )

    return {
        "status": "success",
        "message": "Regression model trained and evaluated!",
        "train_size": len(trainer.x_train),
        "test_size": len(trainer.x_test),
        "num_features": trainer.x_train.shape[1],
        "drop_price_columns": request.drop_price_columns,
        "apply_filter": request.apply_filter,
        "filter_threshold": request.filter_threshold if request.apply_filter else "N/A",
        "mse_filtered": mse_filtered,
        "r2_filtered": r2_filtered,
    }

@app.post("/train-classifiers/")
def train_classifiers():
    global trainer  # ✅ Ensure trainer is globally available

    if trainer is None or trainer.y_test_filtered is None or trainer.predictions_filtered is None:
        return {"status": "error", "message": "Classifier training skipped due to missing regression results."}

    print("📑 Training Classifiers using Regression Predictions...")

    # ✅ Ensure `trainer.x_test_with_meta` contains relevant market data
    if trainer.x_test_with_meta is None:
        return {"status": "error", "message": "Meta columns are missing in x_test_with_meta."}

    # ✅ Ensure the dataset for classification is properly filtered
    try:
        data_selected_filtered = trainer.x_test_with_meta.loc[trainer.y_test_filtered.index].copy()
    except KeyError:
        return {"status": "error", "message": "Filtered indices do not match x_test_with_meta."}

    # ✅ Assign Predicted High
    data_selected_filtered["Predicted_High"] = trainer.predictions_filtered

    # ✅ Shift previous values
    data_selected_filtered["Prev_Close"] = data_selected_filtered["Close"].shift(1)
    data_selected_filtered["Prev_Predicted_High"] = data_selected_filtered["Predicted_High"].shift(1)

    # ✅ Drop NaNs before classification label generation
    data_selected_filtered = data_selected_filtered.dropna(subset=["Predicted_High", "Prev_Close", "Prev_Predicted_High"])

    # ✅ Generate classification labels
    label_gen = LabelGenerator()
    df_with_labels = label_gen.add_good_bar_label(data_selected_filtered)
    if df_with_labels is None or df_with_labels.empty:
        return {"status": "error", "message": "Failed to generate labels for classification."}

    # ✅ Split data for classification
    processor = DataProcessor()
    (
        trainer.classifier_X_train,
        trainer.classifier_y_train,
        trainer.classifier_X_test,
        trainer.classifier_y_test
    ) = processor.prepare_dataset_for_regression_sequential(
        data=df_with_labels.drop(columns=["Predicted_High", "Next_High"], errors="ignore"),
        target_column="good_bar_prediction_outside_of_boundary",
        drop_target=True,
        split_ratio=0.8
    )

    print(f"✅ Classifier Training set: {len(trainer.classifier_X_train)} samples, Test set: {len(trainer.classifier_X_test)} samples")

    # ✅ Initialize the classifier trainer
    classifier_trainer = ClassifierModelTrainer()

    # ✅ Train all three classifiers and store results
    rf_results = classifier_trainer.train_random_forest(
        trainer.classifier_X_train, trainer.classifier_y_train,
        trainer.classifier_X_test, trainer.classifier_y_test
    )

    lgbm_results = classifier_trainer.train_lightgbm(
        trainer.classifier_X_train, trainer.classifier_y_train,
        trainer.classifier_X_test, trainer.classifier_y_test
    )

    xgb_results = classifier_trainer.train_xgboost(
        trainer.classifier_X_train, trainer.classifier_y_train,
        trainer.classifier_X_test, trainer.classifier_y_test
    )

    # ✅ Store results globally (if needed)
    trainer.rf_results = rf_results
    trainer.lgbm_results = lgbm_results
    trainer.xgb_results = xgb_results

    def extract_metrics(results):
        return {
            "accuracy": results["accuracy"],
            "precision_0": results["evaluation_metrics"]["0"]["precision"],
            "recall_0": results["evaluation_metrics"]["0"]["recall"],
            "f1_0": results["evaluation_metrics"]["0"]["f1-score"],
            "precision_1": results["evaluation_metrics"]["1"]["precision"],
            "recall_1": results["evaluation_metrics"]["1"]["recall"],
            "f1_1": results["evaluation_metrics"]["1"]["f1-score"],
        }

    # ✅ Modify return statement in `train_classifiers`
    return {
        "status": "success",
        "message": "Classifier training complete!",
        "classifier_train_size": len(trainer.classifier_X_train),
        "classifier_test_size": len(trainer.classifier_X_test),
        "rf_results": extract_metrics(trainer.rf_results),
        "lgbm_results": extract_metrics(trainer.lgbm_results),
        "xgb_results": extract_metrics(trainer.xgb_results),
    }









if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)