from fastapi import FastAPI, Query
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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






# ✅ Define request model with `drop_price_columns`
class TrainRegressionRequest(BaseModel):
    drop_price_columns: bool = True  # Default to dropping prices

@app.post("/train-regression-model/")
def train_regression_model(request: TrainRegressionRequest):
    global training_df_labels, x_train_regression, y_train_regression, x_test_regression, y_test_regression
    global regression_trained_model, predictions_regression

    if training_df_labels is None:
        return {"status": "error", "message": "Labeled training data is not available. Please generate labels first."}

    # ✅ Initialize Trainer
    trainer = RegressionModelTrainer(include_prices=not request.drop_price_columns)

    # ✅ Prepare Data
    trainer.prepare_data(training_df_labels)

    # ✅ Store training/test data globally
    x_train_regression = trainer.x_train
    y_train_regression = trainer.y_train
    x_test_regression = trainer.x_test
    y_test_regression = trainer.y_test

    # ✅ Train Model
    trainer.train_model()
    regression_trained_model = trainer.model  # Save trained model globally

    # ✅ Make Predictions
    predictions_regression = trainer.make_predictions()

    return {
        "status": "success",
        "message": "Regression model trained and predictions made!",
        "train_size": len(x_train_regression),
        "test_size": len(x_test_regression),
        "num_features": x_train_regression.shape[1],
        "drop_price_columns": request.drop_price_columns
    }




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)