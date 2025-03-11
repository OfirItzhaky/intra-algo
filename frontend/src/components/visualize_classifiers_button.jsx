import React, { useState } from "react";

function VisualizeClassifiersButton() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleClick = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch("http://127.0.0.1:8000/visualize-classifiers/");

            if (!response.ok) {
                throw new Error("Failed to generate classifier visualization.");
            }

            // ✅ Convert response into an image URL and open in a new window
            const blob = await response.blob();
            const imageObjectUrl = URL.createObjectURL(blob);
            window.open(imageObjectUrl, "_blank"); // ✅ Opens in a separate tab like regression

        } catch (err) {
            console.error("🚨 Visualization failed:", err);
            setError("Failed to fetch classifier visualization.");
        }

        setLoading(false);
    };

    return (
        <div>
            <button
                onClick={handleClick}
                disabled={loading}
                style={{
                    backgroundColor: loading ? "#6c757d" : "#007bff",
                    color: "white",
                    padding: "10px 20px",
                    border: "none",
                    borderRadius: "5px",
                    cursor: loading ? "not-allowed" : "pointer",
                    margin: "5px",
                }}
            >
                {loading ? "⏳ Generating..." : "📊 Visualize Classifiers"}
            </button>

            {error && <p style={{ color: "red" }}>⚠️ {error}</p>}
        </div>
    );
}

export default VisualizeClassifiersButton;
