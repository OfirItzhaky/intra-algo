import pandas as pd
from fastapi import FastAPI, Query
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
from fastapi.responses import Response
import matplotlib.pyplot as plt
from fastapi import HTTPException

from backend.new_bar import NewBar
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
# ✅ Store initial simulation data (before adding new bars)
simulation_df_startingpoint = None

first_bar_processed = False  # ✅ Initialize first-bar tracking

@app.get("/load-data/")
def load_data(
    file_path: str = Query(...),
    data_type: str = Query(...),
    symbol: str = Query(...)
):
    global training_df_raw, simulation_df

    base_folder = "data"
    full_path = os.path.join(base_folder, data_type, file_path)

    print(f"🔗 Loading file from: {full_path}")

    df = data_loader.load_from_csv(full_path)

    # ✅ Check if df is empty BEFORE using iloc
    if df.empty or "Date" not in df.columns or "Time" not in df.columns:
        return {"status": "error", "message": f"{data_type.capitalize()} file could not be loaded or is empty."}

    validation_result = None

    if data_type == "training":
        training_df_raw = df  # ✅ Store training data

    elif data_type == "simulating":
        if training_df_raw is None:
            return {"status": "error", "message": "Training data must be loaded before simulation data!"}

        processor = DataProcessor()
        validation_result = processor.validate_simulation(training_df_raw, df)

        # ✅ Apply the fixed simulation data
        simulation_df = validation_result["fixed_simulation_df"]

    # ✅ Extract validation messages
    missing_data_warning = validation_result.get("missing_data_warning", None) if validation_result else None
    insufficient_simulation_warning = validation_result.get("insufficient_simulation_warning", None) if validation_result else None
    overlap_fixed = validation_result.get("overlap_fixed", False) if validation_result else False

    # ✅ Extract first and last rows SAFELY from the correct dataset
    if data_type == "training":
        first_row = training_df_raw.iloc[0] if not training_df_raw.empty else None
        last_row = training_df_raw.iloc[-1] if not training_df_raw.empty else None
    elif data_type == "simulating":
        first_row = simulation_df.iloc[0] if not simulation_df.empty else None
        last_row = simulation_df.iloc[-1] if not simulation_df.empty else None
    else:
        first_row, last_row = None, None  # Fallback case

    summary = {
        "symbol": symbol,
        "first_date": first_row["Date"] if first_row is not None else "N/A",
        "first_time": first_row["Time"] if first_row is not None else "N/A",
        "last_date": last_row["Date"] if last_row is not None else "N/A",
        "last_time": last_row["Time"] if last_row is not None else "N/A",
        "dataType": data_type,
        "missing_data_warning": missing_data_warning,  # ✅ Pass missing data warning
        "insufficient_simulation_warning": insufficient_simulation_warning,  # ✅ Pass simulation too short warning
        "overlap_fixed": overlap_fixed  # ✅ Pass overlap fix status
    }

    print(f"✅ Summary: {summary}")

    return {
        "status": "success",
        "summary": summary
    }




@app.get("/get-loaded-data/")
def get_loaded_data(data_type: str = Query(..., description="Data type - training or simulating")):
    global simulation_df

    if data_type == "training" and training_df_raw is not None:
        return {"status": "success", "data": training_df_raw.head(5).to_dict(orient="records")}

    elif data_type == "simulating" and simulation_df is not None:
        # ✅ Run validation again to ensure simulation data is clean
        processor = DataProcessor()
        validation_result = processor.validate_simulation(training_df_raw, simulation_df)

        return {
            "status": "success",
            "data": validation_result["fixed_simulation_df"].head(5).to_dict(orient="records"),
            "missing_data_warning": validation_result["missing_data_warning"],  # ✅ Pass missing data alerts
            "overlap_fixed": validation_result["overlap_fixed"]  # ✅ Pass overlap fix status
        }

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
    trainer.regression_figure = fig
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
    trainer.regression_figure = processor.visualize_regression_predictions_for_pycharm(
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
    global classifier_trainer  # ✅ Ensure classifier_trainer is globally stored

    if trainer is None or trainer.y_test_filtered is None or trainer.predictions_filtered is None:
        return {"status": "error", "message": "Classifier training skipped due to missing regression results."}

    print("📑 Training Classifiers using Regression Predictions...")

    if trainer.x_test_with_meta is None:
        return {"status": "error", "message": "Meta columns are missing in x_test_with_meta."}

    try:
        data_selected_filtered = trainer.x_test_with_meta.loc[trainer.y_test_filtered.index].copy()
    except KeyError:
        return {"status": "error", "message": "Filtered indices do not match x_test_with_meta."}

    # ✅ Assign Predicted High and shift previous values
    data_selected_filtered["Predicted_High"] = trainer.predictions_filtered
    data_selected_filtered["Prev_Close"] = data_selected_filtered["Close"].shift(1)
    data_selected_filtered["Prev_Predicted_High"] = data_selected_filtered["Predicted_High"].shift(1)

    # ✅ Drop NaNs before classification label generation
    data_selected_filtered = data_selected_filtered.dropna(subset=["Predicted_High", "Prev_Close", "Prev_Predicted_High"])

    # ✅ Generate classification labels
    label_gen = LabelGenerator()
    df_with_labels = label_gen.add_good_bar_label(data_selected_filtered)
    if df_with_labels is None or df_with_labels.empty:
        return {"status": "error", "message": "Failed to generate labels for classification."}

    # ✅ Prepare dataset for classification
    processor = DataProcessor()
    classifier_X_train, classifier_y_train, classifier_X_test, classifier_y_test = processor.prepare_dataset_for_regression_sequential(
        data=df_with_labels.drop(columns=["Predicted_High", "Next_High"], errors="ignore"),
        target_column="good_bar_prediction_outside_of_boundary",
        drop_target=True,
        split_ratio=0.8
    )

    print(f"✅ Classifier Training set: {len(classifier_X_train)} samples, Test set: {len(classifier_X_test)} samples")

    # ✅ Initialize classifier trainer and train all classifiers
    classifier_trainer = ClassifierModelTrainer()
    classifier_trainer.train_all_classifiers(classifier_X_train, classifier_y_train, classifier_X_test, classifier_y_test, trainer)

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

    return {
        "status": "success",
        "message": "Classifier training complete!",
        "classifier_train_size": len(classifier_X_train),
        "classifier_test_size": len(classifier_X_test),
        "classifier_predictions": classifier_trainer.classifier_predictions_df.to_dict(orient="records"),  # ✅ Store predictions
        "rf_results": extract_metrics(classifier_trainer.rf_results),
        "lgbm_results": extract_metrics(classifier_trainer.lgbm_results),
        "xgb_results": extract_metrics(classifier_trainer.xgb_results),
    }



from fastapi.responses import Response
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

@app.get("/debug-regression-figure/")
def debug_regression_figure():
    global trainer  # ✅ Ensure trainer is available

    if trainer is None or trainer.regression_figure is None:
        return {"status": "error", "message": "Regression figure not found. Ensure regression training was completed first."}

    return {"status": "success", "message": "Regression figure exists!"}


@app.get("/visualize-classifiers/")
def visualize_classifiers():
    global classifier_trainer, trainer  # ✅ Ensure both are available

    if trainer is None or classifier_trainer is None:
        return {"status": "error", "message": "Ensure both regression and classifier training are completed."}

    print("📊 Generating Combined Visualization (Regression + Classifiers)...")

    processor = DataProcessor()
    fig = processor.visualize_classifiers_pycharm(
        trainer, classifier_trainer, n=20
    )

    if fig is None:
        return {"status": "error", "message": "Failed to generate visualization."}

    # ✅ Save to memory
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format="png", facecolor="black", bbox_inches="tight")
    plt.close(fig)  # ✅ Prevent memory leaks
    img_bytes.seek(0)

    return Response(content=img_bytes.getvalue(), media_type="image/png")




@app.get("/generate-new-bar/")
def generate_new_bar(validate: bool = False):
    global simulation_df, regression_trained_model, classifier_trainer
    global first_bar_processed  # ✅ Ensure it’s accessed globally

    if simulation_df is None or simulation_df.empty:
        raise HTTPException(status_code=400, detail="Simulation data is not loaded.")

    # ✅ Extract next available bar
    next_bar_data = simulation_df.iloc[0].to_dict()  # Take first row
    simulation_df.drop(index=simulation_df.index[0], inplace=True)  # Remove from queue

    # ✅ First bar alignment validation (ONLY for first bar)
    if not first_bar_processed:
        # 🔹 Extract the last two timestamps from classifier predictions DataFrame
        last_two_rows = classifier_trainer.classifier_predictions_df.iloc[-2:]  # Get last two rows

        # ✅ Extract the index values since the Date/Time are in the index
        last_timestamp = pd.to_datetime(last_two_rows.index[-1])
        second_last_timestamp = pd.to_datetime(last_two_rows.index[-2])

        # ✅ Compute expected interval
        expected_interval = last_timestamp - second_last_timestamp

        # ✅ Compute expected first timestamp dynamically
        expected_first_timestamp = last_timestamp + expected_interval

        # ✅ Get actual first timestamp from simulation data
        actual_first_timestamp = pd.to_datetime(next_bar_data["Date"] + " " + next_bar_data["Time"])

        # ✅ Validate the timestamp
        if actual_first_timestamp != expected_first_timestamp:
            raise HTTPException(status_code=400,
                                detail=f"First bar timestamp mismatch! Expected '{expected_first_timestamp}', but got '{actual_first_timestamp}'")

        first_bar_processed = True  # ✅ Mark as processed

    # ✅ Create NewBar instance and compute features
    new_bar = NewBar(**next_bar_data)
    new_bar._1_calculate_indicators_new_bar(simulation_df)
    new_bar._2_add_vwap_new_bar(simulation_df)
    new_bar._3_add_fibonacci_levels_new_bar(simulation_df)
    new_bar._4_add_cci_average_new_bar(simulation_df)
    new_bar._5_add_ichimoku_cloud_new_bar(simulation_df)
    new_bar._6_add_atr_price_features_new_bar(simulation_df)
    new_bar._7_add_multi_ema_indicators_new_bar(simulation_df)
    new_bar._8_add_high_based_indicators_combined_new_bar(simulation_df)
    new_bar._9_add_constant_columns_new_bar()
    new_bar._10_add_macd_indicators_new_bar(simulation_df)
    new_bar._11_add_volatility_momentum_volume_features_new_bar(simulation_df)

    # ✅ Existing NaN validation (unchanged)
    if validate:
        new_bar.validate_new_bar()

    # ✅ Convert to DataFrame format for model input
    new_bar_df = pd.DataFrame([vars(new_bar)])

    # ✅ Run regression model
    predicted_high = regression_trained_model.predict(
        new_bar_df.drop(columns=["Date", "Time", "Open", "High", "Low", "Close", "Volume"], errors="ignore"))[0]
    new_bar_df["Predicted_High"] = predicted_high

    # ✅ Prepare classifier input
    new_bar_df["Prev_Close"] = new_bar_df["Close"].shift(1)
    new_bar_df["Prev_Predicted_High"] = new_bar_df["Predicted_High"].shift(1)
    new_bar_df.dropna(subset=["Prev_Close", "Prev_Predicted_High"], inplace=True)

    # ✅ Run classifiers
    classifier_predictions = classifier_trainer.predict_all_classifiers(
        new_bar_df.drop(columns=["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Predicted_High"],
                        errors="ignore"))

    # ✅ Attach predictions
    new_bar_df["RandomForest"] = classifier_predictions["RandomForest"]
    new_bar_df["LightGBM"] = classifier_predictions["LightGBM"]
    new_bar_df["XGBoost"] = classifier_predictions["XGBoost"]

    return {
        "status": "success",
        "new_bar": new_bar_df.iloc[0].to_dict()
    }



@app.get("/initialize-simulation/")
def initialize_simulation():
    global simulation_df_startingpoint

    # ✅ Ensure all required data is available
    if trainer is None or classifier_trainer is None:
        return {"status": "error", "message": "Regression and classifier models are not initialized!"}

    if trainer.x_test_with_meta is None or trainer.y_test is None:
        return {"status": "error", "message": "No test data available for simulation!"}

    if classifier_trainer.classifier_predictions_df is None:
        return {"status": "error", "message": "Classifier predictions are missing!"}

    # ✅ Extract relevant columns from regression data
    regression_data = trainer.x_test_with_meta.loc[trainer.y_test.index].copy()
    regression_data["Actual_High"] = trainer.y_test.values
    regression_data["Predicted_High"] = trainer.predictions

    # ✅ Merge Date and Time for proper alignment
    meta_columns = training_df_labels[["Date", "Time", "Open", "High", "Low", "Close"]]
    # ✅ Drop overlapping columns from `meta_columns` before joining
    meta_columns = meta_columns.drop(columns=["Date", "Time", "Open", "High", "Low", "Close"], errors="ignore")

    # ✅ Now join without duplicate errors
    regression_data = regression_data.join(meta_columns, how="left")

    # ✅ Create a proper Timestamp column (for merging with classifiers)
    regression_data["Timestamp"] = pd.to_datetime(regression_data["Date"] + " " + regression_data["Time"])
    regression_data.set_index("Timestamp", inplace=True)

    # ✅ Ensure classifier predictions match timestamp format
    classifier_predictions = classifier_trainer.classifier_predictions_df.copy()
    classifier_predictions.index = pd.to_datetime(classifier_predictions.index)  # Convert index

    # ✅ Merge regression + classifier predictions
    simulation_df_startingpoint = regression_data.merge(
        classifier_predictions, left_index=True, right_index=True, how="left"
    )

    # ✅ Keep only necessary columns
    simulation_df_startingpoint = simulation_df_startingpoint[
        ["Date", "Time", "Open", "High", "Low", "Close", "Actual_High", "Predicted_High", "RandomForest", "LightGBM", "XGBoost"]
    ]
    # ✅ Drop rows where any classifier prediction is NaN
    simulation_df_startingpoint = simulation_df_startingpoint.dropna(subset=["RandomForest", "LightGBM", "XGBoost"])
    # ✅ Check for NaNs in any other columns (excluding classifier predictions)
    other_columns_with_nans = simulation_df_startingpoint.drop(
        columns=["RandomForest", "LightGBM", "XGBoost"]).isna().sum()

    # ✅ If any NaNs remain in other columns, raise an error
    if other_columns_with_nans.sum() > 0:
        raise ValueError(f"❌ Unexpected NaNs found in non-classifier columns:\n{other_columns_with_nans}")

    # ✅ Debugging Output
    print("✅ Final Simulation Starting Point DF:")
    print(simulation_df_startingpoint.head())

    return {
        "status": "success",
        "data": simulation_df_startingpoint.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)