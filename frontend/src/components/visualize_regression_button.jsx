import React, { useState } from 'react';

function VisualizeRegressionButton() {
    const [loading, setLoading] = useState(false);

    const handleVisualization = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://localhost:8000/get-regression-chart");
            if (!response.ok) throw new Error("Failed to fetch regression visualization.");

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            window.open(imageUrl, "_blank"); // ✅ Open in a new tab
        } catch (error) {
            console.error("🚨 Error fetching regression chart:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <button className="visualize-regression-button" onClick={handleVisualization} disabled={loading}>
            {loading ? "Loading..." : "Visualize Regression"}
        </button>
    );
}

export default VisualizeRegressionButton;
