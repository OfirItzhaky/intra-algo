import React, { useState } from "react";

function TrainClassifiersButton({ onClassificationComplete }) {  // ✅ Fix prop name to match App.jsx
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleClick = async () => {
    setLoading(true);
    setError(null);

    try {
        const response = await fetch("http://127.0.0.1:8000/train-classifiers/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        const result = await response.json();

        if (result.status === "success") {
            console.log("✅ Classifiers successfully trained!", result);

            // ✅ Fix: Correctly pass classifier results
            onClassificationComplete({
                RandomForest: result.rf_results,
                LightGBM: result.lgbm_results,
                XGBoost: result.xgb_results
            });
        } else {
            console.error("⚠️ Classifier training failed:", result.message);
            setError(result.message);
        }
    } catch (err) {
        console.error("🚨 API request failed:", err);
        setError("Failed to connect to the server.");
    }

    setLoading(false);
};


    return (
        <div>
            <button
                onClick={handleClick}
                disabled={loading}
                style={{
                    backgroundColor: loading ? "#6c757d" : "#dc3545",
                    color: "white",
                    padding: "10px 20px",
                    border: "none",
                    borderRadius: "5px",
                    cursor: loading ? "not-allowed" : "pointer",
                    margin: "5px",
                }}
            >
                {loading ? "⏳ Training..." : "📑 Train Classifiers"}
            </button>

            {error && <p style={{ color: "red" }}>⚠️ {error}</p>}
        </div>
    );
}

export default TrainClassifiersButton;
