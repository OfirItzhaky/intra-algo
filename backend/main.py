import pandas as pd
from fastapi import FastAPI, Query
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import HTTPException
import json
from new_bar import NewBar
from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor
from regression_model_trainer import RegressionModelTrainer
from label_generator import LabelGenerator
from feature_generator import FeatureGenerator
from data_loader import DataLoader
import re
from imblearn.over_sampling import SMOTE

from fastapi.responses import Response
import matplotlib.pyplot as plt
from io import BytesIO
app = FastAPI()


# CORS for frontend
origins = [f"http://localhost:{port}" for port in range(5173, 5184)]
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
simulation_df_original = None
first_bar_processed = False  # ✅ Initialize first-bar tracking
regression_historical_data = None
classifier_historical_data = None
@app.get("/load-data/")
def load_data(
    file_path: str = Query(...),
    data_type: str = Query(...),
    symbol: str = Query(...)
):
    global training_df_raw, simulation_df, simulation_df_original  # ✅ Added `simulation_df_original`

    base_folder = "data"
    full_path = os.path.join(base_folder, data_type, file_path)

    print(f"🔗 Loading file from: {full_path}")

    df = data_loader.load_from_csv(full_path)

    # ✅ Check if df is empty BEFORE using iloc
    if df.empty or "Date" not in df.columns or "Time" not in df.columns:
        return {"status": "error", "message": f"{data_type.capitalize()} file could not be loaded or is empty."}

    df = df.drop_duplicates(subset=["Date", "Time"])

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

        # ✅ Store a **backup copy** ONLY if `simulation_df_original` is not already set
        if simulation_df_original is None:
            simulation_df_original = simulation_df.copy()  # ✅ Store original for restarting

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
        # ✅ Write demo bars to JS file at simulation load
        demo_js_content = """export let initialData = [
          { "date": "2020-01-01 00:00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5 },
          { "date": "2020-01-01 00:05:00", "open": 1.5, "high": 2.0, "low": 1.0, "close": 1.8 },
          { "date": "2020-01-01 00:10:00", "open": 1.8, "high": 2.2, "low": 1.6, "close": 2.0 }
        ];

        export let PredActualData = [
          { "date": "2020-01-01 00:00:00", "actualHigh": 2.0, "predictedHigh": 2.1 },
          { "date": "2020-01-01 00:05:00", "actualHigh": 2.0, "predictedHigh": 2.05 },
          { "date": "2020-01-01 00:10:00", "actualHigh": 2.2, "predictedHigh": 2.3 }
        ];

        export let classifierData = [
          { "date": "2020-01-01 00:00:00", "rf": 1.0, "lt": 0.0, "xg": 1.0 },
          { "date": "2020-01-01 00:05:00", "rf": 0.0, "lt": 1.0, "xg": 1.0 },
          { "date": "2020-01-01 00:10:00", "rf": 1.0, "lt": 1.0, "xg": 0.0 }
        ];
        """
        with open("../frontend/src/initialData.js", "w") as js_file:
            js_file.write(demo_js_content)
        print("🔄 Demo JS file initialized with 3 bars.")

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


    processor = DataProcessor()
    fig = processor.visualize_regression_predictions_for_pycharm(
        trainer.x_test_with_meta, trainer.y_test, trainer.predictions, n=20  # Default to 1,000 for scrolling
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
        # ✅ Store valid indices globally for classifier training
        global valid_prediction_indices
        valid_prediction_indices = valid_indices  # ✅ Save filtered indices globally
    else:
        # ✅ No filtering: Use all predictions for evaluation
        trainer.y_test_filtered = trainer.y_test.copy()
        trainer.predictions_filtered = trainer.predictions.copy()

    # ✅ Compute Metrics on Filtered Data
    mse_filtered = mean_squared_error(trainer.y_test_filtered, trainer.predictions_filtered)
    r2_filtered = r2_score(trainer.y_test_filtered, trainer.predictions_filtered)


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
@app.post("/generate-classifier-labels/")
def generate_classifier_labels(label_method: str = Query(..., description="Label method to use")):
    global trainer, labeled_data_for_training, classifier_labels_generated
    
    # Initialize the flag if it doesn't exist
    if 'classifier_labels_generated' not in globals():
        classifier_labels_generated = False
    
    if trainer is None or trainer.y_test_filtered is None or trainer.predictions_filtered is None:
        return {"status": "error", "message": "Classifier label generation skipped due to missing regression results."}

    print(f"🏷️ Generating Classifier Labels using method: {label_method}")

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

    # ✅ Generate selected classification labels
    label_gen = LabelGenerator()
    
    # Apply the selected labeling method
    if label_method == "add_good_bar_label":
        df_with_labels = label_gen.add_good_bar_label(data_selected_filtered)
        label_column = "good_bar_prediction_outside_of_boundary"
    elif label_method == "long_good_bar_label_all":
        df_with_labels = label_gen.long_good_bar_label_all(data_selected_filtered)
        label_column = "long_good_bar_label"
    elif label_method == "long_good_bar_label_bullish_only":
        df_with_labels = label_gen.long_good_bar_label_bullish_only(data_selected_filtered)
        label_column = "long_good_bar_label"
    elif label_method == "long_good_bar_label_all_goal_c":
        df_with_labels = label_gen.long_good_bar_label_all_goal_c(data_selected_filtered)
        label_column = "long_good_bar_label"
    elif label_method == "long_good_bar_label_bullish_only_goal_c":
        df_with_labels = label_gen.long_good_bar_label_bullish_only_goal_c(data_selected_filtered)
        label_column = "long_good_bar_label"
    else:
        return {"status": "error", "message": f"Invalid labeling method: {label_method}"}
    
    if df_with_labels is None or df_with_labels.empty:
        return {"status": "error", "message": "Failed to generate labels for classification."}

    # Store the labeled data for training later
    labeled_data_for_training = df_with_labels.copy()
    
    # Set the flag indicating labels have been generated
    classifier_labels_generated = True
    
    # Also store which label method and column were used
    global classifier_label_method, classifier_label_column
    classifier_label_method = label_method
    classifier_label_column = label_column
    
    # Calculate some statistics about the generated labels
    total_labels = len(df_with_labels)
    positive_labels = df_with_labels[label_column].sum()
    positive_percentage = (positive_labels / total_labels) * 100
    
    return {
        "status": "success",
        "message": f"Successfully generated {label_method} labels",
        "total_labels": total_labels,
        "positive_labels": int(positive_labels),
        "positive_percentage": round(positive_percentage, 2),
        "label_column": label_column,
        "label_method": label_method
    }

@app.post("/train-classifiers/")
def train_classifiers():
    global classifier_trainer, labeled_data_for_training, classifier_labels_generated
    
    # Initialize the flag if it doesn't exist
    if 'classifier_labels_generated' not in globals():
        classifier_labels_generated = False
    
    # Check if labels have been generated
    if not classifier_labels_generated or labeled_data_for_training is None:
        return {
            "status": "error", 
            "message": "Please generate classifier labels first using the 'Generate Classifier Label' button!"
        }

    if trainer is None or trainer.y_test_filtered is None or trainer.predictions_filtered is None:
        return {"status": "error", "message": "Classifier training skipped due to missing regression results."}

    print("📑 Training Classifiers using pre-generated labels...")
    
    # Use the pre-generated labels
    df_with_labels = labeled_data_for_training
    target_column = classifier_label_column
    
    # ✅ Prepare dataset for classification
    processor = DataProcessor()
    classifier_X_train, classifier_y_train, classifier_X_test, classifier_y_test = processor.prepare_dataset_for_regression_sequential(
        data=df_with_labels.drop(columns=["Predicted_High", "Next_High"], errors="ignore"),
        target_column=target_column,
        drop_target=True,
        split_ratio=0.8
    )

    print(f"✅ Classifier Training set: {len(classifier_X_train)} samples, Test set: {len(classifier_X_test)} samples")
    print(f"✅ Using target column: {target_column} from method: {classifier_label_method}")

    # ✅ Apply SMOTE to the training data only
    smote = SMOTE(random_state=42)
    classifier_X_train_bal, classifier_y_train_bal = smote.fit_resample(classifier_X_train, classifier_y_train)

    print(f"✅ SMOTE applied: Balanced training size = {len(classifier_X_train_bal)}")
    # ✅ Initialize classifier trainer and train all classifiers
    classifier_trainer = ClassifierModelTrainer()
    # ✅ Extract timestamps separately
    meta_timestamps = df_with_labels[["Date", "Time"]].copy()

    classifier_trainer.train_all_classifiers(
        classifier_X_train_bal, classifier_y_train_bal, classifier_X_test, classifier_y_test,
        meta_timestamps_df=meta_timestamps
    )

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
    # ✅ Run and store cross-val SMOTE metrics
    X_all = pd.concat([classifier_X_train_bal, classifier_X_test])
    y_all = pd.concat([classifier_y_train_bal, classifier_y_test])
    cv_smote_results = classifier_trainer.evaluate_with_cross_val_smote(X_all, y_all, label="Current")

    # ✅ Extract individual CV results from list of dicts
    cv_rf_result = next((item for item in cv_smote_results if item["Model"] == "RandomForest"), {})
    cv_lgbm_result = next((item for item in cv_smote_results if item["Model"] == "LightGBM"), {})
    cv_xgb_result = next((item for item in cv_smote_results if item["Model"] == "XGBoost"), {})

    # Reset flags after successful training
    classifier_labels_generated = False
    labeled_data_for_training = None

    return {
        "status": "success",
        "message": f"Classifier training complete using '{classifier_label_method}' labels!",
        "classifier_train_size": len(classifier_X_train_bal),
        "classifier_test_size": len(classifier_X_test),
        "classifier_predictions": classifier_trainer.classifier_predictions_df.to_dict(orient="records"),
        "target_column": target_column,
        "label_method": classifier_label_method,

        # 🔵 Non-CV Metrics
        "rf_results": extract_metrics(classifier_trainer.rf_results),
        "lgbm_results": extract_metrics(classifier_trainer.lgbm_results),
        "xgb_results": extract_metrics(classifier_trainer.xgb_results),

        # 🟣 Cross-Validation Metrics (with SMOTE)
        "cv_rf_results": cv_rf_result,
        "cv_lgbm_results": cv_lgbm_result,
        "cv_xgb_results": cv_xgb_result,

        # Optional full table for export/comparison
        "cv_results_df": cv_smote_results  # optional for rendering if needed
    }




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
    global simulation_df, regression_trained_model, classifier_trainer, trainer
    global newbar_created_df_for_simulator  # ✅ Stores newly created bars for future use
    global regression_historical_data , classifier_historical_data
    global first_bar_processed

    if simulation_df is None or simulation_df.empty:
        raise HTTPException(status_code=400, detail="Simulation data is not loaded.")

    # ✅ Ensure global structures exist
    if 'newbar_created_df_for_simulator' not in globals() or newbar_created_df_for_simulator is None:
        newbar_created_df_for_simulator = pd.DataFrame()



    next_bar_data = simulation_df.iloc[0].to_dict()  # Take first row
    simulation_df.drop(index=simulation_df.index[0], inplace=True)  # Remove from queue



    # ✅ Create NewBar instance with explicit fields
    new_bar = NewBar(
        Open=next_bar_data["Open"],
        High=next_bar_data["High"],
        Low=next_bar_data["Low"],
        Close=next_bar_data["Close"],
        Volume=next_bar_data["Volume"],
        Date=next_bar_data["Date"],
        Time=next_bar_data["Time"]
    )

    # ✅ If this is the first bar, initialize from x_test_with_meta
    if not first_bar_processed:
        regression_historical_data = trainer.x_test_with_meta.copy()

    # ✅ Ensure historical data contains all necessary features before calculations
    if regression_historical_data is None or regression_historical_data.empty:
        raise ValueError(
            "❌ `regression_historical_data` is empty! Ensure it is initialized before calling indicators.")

    new_bar._1_calculate_indicators_new_bar(historical_data=regression_historical_data)
    new_bar._2_add_vwap_new_bar(historical_data=regression_historical_data)
    new_bar._3_add_fibonacci_levels_new_bar(historical_data=regression_historical_data)
    new_bar._4_add_cci_average_new_bar(historical_data=regression_historical_data)
    new_bar._5_add_ichimoku_cloud_new_bar(historical_data=regression_historical_data)
    new_bar._6_add_atr_price_features_new_bar(historical_data=regression_historical_data)
    new_bar._7_add_multi_ema_indicators_new_bar(historical_data=regression_historical_data)
    new_bar._8_add_high_based_indicators_combined_new_bar(historical_data=regression_historical_data)
    new_bar._9_add_constant_columns_new_bar()
    new_bar._10_add_macd_indicators_new_bar(historical_data=regression_historical_data)
    new_bar._11_add_volatility_momentum_volume_features_new_bar(historical_data=regression_historical_data)

    # ✅ Validate New Bar
    if validate:
        new_bar.validate_new_bar()

    # ✅ Append `new_bar` to regression historical data for future use
    new_bar_df = pd.DataFrame([vars(new_bar)])  # Convert to DataFrame
    regression_historical_data = pd.concat([regression_historical_data, new_bar_df],
                                           ignore_index=True)  # ✅ Append to history
    regression_historical_data = regression_historical_data.drop_duplicates(subset=["Date", "Time"],
                                                                            keep="first")  # ✅ Avoid duplicate rows
    # ✅ Run regression model
    predicted_high = regression_trained_model.predict(
        regression_historical_data.iloc[[-1]][regression_trained_model.feature_names_in_].copy()
    )[0]

    # ✅ Attach prediction to new_bar
    new_bar.Predicted_High = predicted_high
    new_bar_df["Predicted_High"] = predicted_high


    # ✅ If this is the first bar, initialize from x_test_with_meta
    if not first_bar_processed:
        new_bar_df["Prev_Close"] = regression_historical_data.iloc[-2]["Close"]  # ✅ Corrected
        new_bar_df["Prev_Predicted_High"] = trainer.predictions[-1]  # ✅ Use last known prediction
    else:
        new_bar_df["Prev_Close"] = classifier_historical_data.iloc[-1]["Close"]  # ✅ Keep using regression history
        new_bar_df["Prev_Predicted_High"] = classifier_historical_data.iloc[-1][
            "Predicted_High"]  # ✅ Use last classifier prediction

    # ✅ Run classifiers
    classifier_predictions = classifier_trainer.predict_all_classifiers(
        new_bar_df[classifier_trainer.rf_results["model"].feature_names_in_].copy()
    )

    # ✅ Attach classifier predictions to new_bar
    new_bar.RandomForest = classifier_predictions["RandomForest"]
    new_bar.LightGBM = classifier_predictions["LightGBM"]
    new_bar.XGBoost = classifier_predictions["XGBoost"]

    # ✅ Attach classifier predictions to new_bar_df for UI response
    new_bar_df["RandomForest"] = classifier_predictions["RandomForest"]
    new_bar_df["LightGBM"] = classifier_predictions["LightGBM"]
    new_bar_df["XGBoost"] = classifier_predictions["XGBoost"]

    # ✅ Log Output for Debugging
    print("\n📊 **New Bar Processed Successfully:**")
    print(new_bar_df[["Date", "Time", "Predicted_High", "RandomForest", "LightGBM", "XGBoost"]].to_string(index=False))

    classifier_historical_data = pd.concat([classifier_historical_data, new_bar_df],
                                           ignore_index=True)  # ✅ Append to history
    classifier_historical_data = classifier_historical_data.drop_duplicates(subset=["Date", "Time"],
                                                                            keep="first")  # ✅ Avoid duplicate rows

    # Read the file
    file_path = os.path.join("..", "frontend", "src", "components", "initialData.js")
    with open(file_path, "r") as f:
        content = f.read()

    # Extract JSON strings using regex
    initial_data_str = re.search(r"export let initialData = (\[.*?\]);", content, re.DOTALL).group(1)
    pred_actual_data_str = re.search(r"export let PredActualData = (\[.*?\]);", content, re.DOTALL).group(1)
    classifier_data_str = re.search(r"export let classifierData = (\[.*?\]);", content, re.DOTALL).group(1)

    # Parse JSON
    initialData = json.loads(initial_data_str)
    PredActualData = json.loads(pred_actual_data_str)
    classifierData = json.loads(classifier_data_str)

    PredActualData[-1]["actualHigh"] = float(new_bar.High)

    # Now you can append new data to these lists
    initialData.append({
        "date": f"{new_bar.Date} {new_bar.Time}",
        "open": float(new_bar.Open),
        "high": float(new_bar.High),
        "low": float(new_bar.Low),
        "close": float(new_bar.Close)
    })

    PredActualData.append({
        "date": f"{new_bar.Date} {new_bar.Time}",
        "actualHigh": None,
        "predictedHigh": float(new_bar.Predicted_High)
    })

    classifierData.append({
        "date": f"{new_bar.Date} {new_bar.Time}",
        "rf": int(new_bar.RandomForest),
        "lt": int(new_bar.LightGBM),
        "xg": int(new_bar.XGBoost)
    })

    # Save updated data
    with open(file_path, "w") as f:
        f.write(f"export let initialData = {json.dumps(initialData, indent=2)};\n")
        f.write(f"export let PredActualData = {json.dumps(PredActualData, indent=2)};\n")
        f.write(f"export let classifierData = {json.dumps(classifierData, indent=2)};\n")
        
    
    first_bar_processed = True

    return {
        "status": "success",
        "new_bar": {
            "date": f"{new_bar.Date} {new_bar.Time}",
            "open": float(new_bar.Open),
            "high": float(new_bar.High),
            "low": float(new_bar.Low),
            "close": float(new_bar.Close),
            "volume": int(new_bar.Volume),
            "actualHigh": None,
            "predictedHigh": float(new_bar.Predicted_High),
            "rf": int(new_bar.RandomForest),
            "lt": int(new_bar.LightGBM),
            "xg": int(new_bar.XGBoost),
        }
    }



@app.get("/initialize-simulation/")
def initialize_simulation():
    global simulation_df_startingpoint, simulation_df_original, first_bar_processed, trainer, classifier_trainer

    # ✅ Ensure all required data is available
    if trainer is None or classifier_trainer is None:
        return {"status": "error", "message": "Regression and classifier models are not initialized!"}

    if trainer.x_test_with_meta is None or trainer.y_test is None:
        return {"status": "error", "message": "No test data available for simulation!"}

    if classifier_trainer.classifier_predictions_df is None:
        return {"status": "error", "message": "Classifier predictions are missing!"}
    first_bar_processed = False

    # ✅ Extract relevant columns from regression data
    regression_data = trainer.x_test_with_meta.loc[trainer.y_test.index].copy()
    regression_data["Actual_High"] = trainer.y_test.values
    regression_data["Predicted_High"] = trainer.predictions

    # ✅ Remove overlapping columns from `meta_columns`
    meta_columns = training_df_labels[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
    meta_columns = meta_columns.drop(columns=["Date", "Time", "Open", "High", "Low", "Close"],
                                     errors="ignore")  # ✅ Drop duplicates

    # ✅ Join the cleaned meta_columns to regression_data
    regression_data = regression_data.join(meta_columns, how="left", rsuffix="_meta")

    # ✅ Create timestamp column
    regression_data["Timestamp"] = pd.to_datetime(regression_data["Date"] + " " + regression_data["Time"])
    regression_data.set_index("Timestamp", inplace=True)

    # ✅ Ensure classifier predictions match timestamp format
    classifier_predictions = classifier_trainer.classifier_predictions_df.copy()
    classifier_predictions.index = pd.to_datetime(classifier_predictions.index)

    simulation_df_startingpoint = regression_data.merge(
        classifier_predictions, left_index=True, right_index=True, how="left"
    )

    # ✅ Keep only necessary columns
    simulation_df_startingpoint = simulation_df_startingpoint[
        ["Date", "Time", "Open", "High", "Low", "Close", "Actual_High", "Predicted_High", "RandomForest", "LightGBM", "XGBoost"]
    ]

    # ✅ Drop NaNs
    simulation_df_startingpoint = simulation_df_startingpoint.dropna(subset=["RandomForest", "LightGBM", "XGBoost"])
    # ✅ Check for NaNs in any other columns (excluding classifier predictions)
    other_columns_with_nans = simulation_df_startingpoint.drop(
        columns=["RandomForest", "LightGBM", "XGBoost"]).isna().sum()

    # ✅ If any NaNs remain in other columns, raise an error
    if other_columns_with_nans.sum() > 0:
        raise ValueError(f"❌ Unexpected NaNs found in non-classifier columns:\n{other_columns_with_nans}")

    print("✅ Final Simulation Starting Point DF:")
    print(simulation_df_startingpoint.head())
    # Step 1: Add date field
    simulation_df_startingpoint["date"] = pd.to_datetime(
        simulation_df_startingpoint["Date"] + " " + simulation_df_startingpoint["Time"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Step 2: Prepare subsets
    initial_data = simulation_df_startingpoint[["date", "Open", "High", "Low", "Close"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    pred_actual_data = simulation_df_startingpoint[["date", "Actual_High", "Predicted_High"]].rename(
        columns={"Actual_High": "actualHigh", "Predicted_High": "predictedHigh"}
    )
    classifier_data = simulation_df_startingpoint[["date", "RandomForest", "LightGBM", "XGBoost"]].rename(
        columns={"RandomForest": "rf", "LightGBM": "lt", "XGBoost": "xg"}
    )

    # Step 3: Combine to JS string
    js_lines = [
        f"export let initialData = {json.dumps(initial_data.to_dict(orient='records'), indent=2)};\n",
        f"export let PredActualData = {json.dumps(pred_actual_data.to_dict(orient='records'), indent=2)};\n",
        f"export let classifierData = {json.dumps(classifier_data.to_dict(orient='records'), indent=2)};\n"
    ]

    # Step 4: Write to file
    output_path = os.path.join("..", "frontend", "src", "components", "initialData.js")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.writelines(js_lines)

    return {
        "status": "success",
        "data": simulation_df_startingpoint.to_dict(orient="records"),
    }

@app.get("/restart-simulation/")
def restart_simulation():
    global simulation_df, simulation_df_original, first_bar_processed

    if simulation_df_original is None:
        return {"status": "error", "message": "Cannot restart simulation. No backup available."}

    # ✅ Restore the simulation data from the original copy
    simulation_df = simulation_df_original.copy()

    # ✅ Reset first bar tracking
    first_bar_processed = False

    # ✅ Call initialize_simulation to reset the initialData.js file
    initialize_simulation()

    print("🔄 Simulation restarted successfully! Simulation data and initialData.js restored.")

    return {
        "status": "success",
        "message": "Simulation restarted to initial state and initialData.js reset."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

  