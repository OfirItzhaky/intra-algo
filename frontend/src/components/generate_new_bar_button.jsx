import React, { useState } from "react";
import axios from "axios";

function GenerateNewBarButton({ onNewBarGenerated, isFirstBarGenerated, setIsFirstBarGenerated }) {
    const [loading, setLoading] = useState(false);

    const handleGenerateNewBar = async () => {
        if (loading) return; // Prevent multiple clicks
        setLoading(true);

        try {
            const response = await axios.get("http://localhost:8000/generate-new-bar/");

            if (response.data.status === "success") {
                const newBar = response.data.new_bar;

                if (!isFirstBarGenerated) {
                    // ✅ First bar: No actual value yet
                    setIsFirstBarGenerated(true);
                    onNewBarGenerated({ ...newBar, actual: null }); // Ensure actual value is missing
                } else {
                    // ✅ Subsequent bars: Attach actual value of previous bar
                    onNewBarGenerated(newBar);
                }
            } else {
                console.error("Error generating new bar:", response.data.message);
            }
        } catch (error) {
            console.error("Request failed:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <button onClick={handleGenerateNewBar} disabled={loading} className="generate-bar-button">
            {loading ? "Generating..." : "➕ Generate Next Bar"}
        </button>
    );
}

export default GenerateNewBarButton;
