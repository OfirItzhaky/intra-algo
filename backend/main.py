# backend/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from data_loader import DataLoader

app = FastAPI()

# Allow frontend to talk to backend
origins = ["http://localhost:5173"]  # Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create instance of DataLoader
data_loader = DataLoader()


@app.get("/load-data/")
def load_data(file_path: str = Query(..., description="Path to data file")):
    try:
        df = data_loader.load_from_csv(file_path)
        if df.empty:
            return {"status": "error", "message": "File could not be loaded or is empty."}

        # Just return first 5 rows for now
        return {
            "status": "success",
            "data": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
