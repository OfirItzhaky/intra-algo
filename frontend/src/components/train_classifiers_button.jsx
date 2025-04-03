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

                // ✅ Pass both standard and CV results
                onClassificationComplete({
                    // Standard metrics
                    RandomForest: result.rf_results,
                    LightGBM: result.lgbm_results,
                    XGBoost: result.xgb_results,
                    // Cross-validation metrics
                    cv_results: {
                        RandomForest: result.cv_rf_results,
                        LightGBM: result.cv_lgbm_results,
                        XGBoost: result.cv_xgb_results
                    }
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
        <div style={{ display: 'flex', flexDirection: 'column', margin: '5px' }}>
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
                }}
            >
                {loading ? "⏳ Training..." : "📑 Train Classifiers"}
            </button>

            {error && (
                <div style={{ 
                    color: '#f87171', 
                    marginTop: '5px', 
                    fontSize: '0.9em',
                    backgroundColor: 'rgba(248, 113, 113, 0.1)',
                    padding: '5px',
                    borderRadius: '5px' 
                }}>
                    ⚠️ {error}
                </div>
            )}
        </div>
    );
}

export default TrainClassifiersButton;
