from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os

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
training_df = None
simulation_df = None

@app.get("/load-data/")
def load_data(
    file_path: str = Query(...),
    data_type: str = Query(...),
    symbol: str = Query(...)
):
    global training_df, simulation_df

    base_folder = "data"
    full_path = os.path.join(base_folder, data_type, file_path)

    print(f"🔗 Loading file from: {full_path}")

    df = data_loader.load_from_csv(full_path)
    if df.empty:
        return {"status": "error", "message": f"{data_type.capitalize()} file could not be loaded or is empty."}

    if data_type == "training":
        training_df = df

    elif data_type == "simulating":
        if training_df is None:
            return {"status": "error", "message": "Training data must be loaded before simulation data!"}

        try:
            df = data_loader.align_and_validate_simulation(training_df, df)
            simulation_df = df  # Save after alignment
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
    if data_type == "training" and training_df is not None:
        return {"status": "success", "data": training_df.head(5).to_dict(orient="records")}
    elif data_type == "simulating" and simulation_df is not None:
        return {"status": "success", "data": simulation_df.head(5).to_dict(orient="records")}
    else:
        return {"status": "error", "message": f"No {data_type} data loaded yet."}


@app.post("/generate-features/")
def generate_features():
    global training_df

    # Ensure training data exists before feature generation
    if training_df is None:
        return {"status": "error", "message": "Training data is not loaded. Please load data first."}

    # Run feature generation
    feature_generator = FeatureGenerator()
    df_features = feature_generator.create_all_features(training_df)

    # Count the number of new features
    original_cols = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
    num_new_features = len(set(df_features.columns) - original_cols)

    # Store the enhanced DataFrame in memory
    training_df = df_features

    return {
        "status": "success",
        "message": "Feature generation completed successfully!",
        "new_features_count": num_new_features
    }

label_generator = LabelGenerator()

@app.get("/generate-labels/")
def generate_labels(label_type: str = Query(..., description="Label type (next_high or good_bar)")):
    global training_df  # Ensure we're modifying the global variable

    if training_df is None or training_df.empty:
        return {"status": "error", "message": "Training data must be loaded before generating labels."}

    if label_type == "next_high":
        training_df = label_generator.elasticnet_label_next_high(training_df)  # ✅ Pass DF and get updated DF
    elif label_type == "good_bar":
        training_df = label_generator.add_good_bar_label(training_df)  # ✅ Pass DF and get updated DF
    else:
        return {"status": "error", "message": "Invalid label type selected."}

    rows_labeled = len(training_df)  # ✅ Count rows in updated DataFrame

    return {
        "status": "success",
        "summary": {
            "label_type": label_type,
            "rows_labeled": rows_labeled
        }
    }

