import React, { useState } from 'react';

function StartSimulationButton() {
    const [loading, setLoading] = useState(false);

    const handleClick = async () => {
        if (loading) return;

        setLoading(true);
        console.log('🚀 Initializing and Opening Simulation...');

        try {
            // First call the backend to initialize simulation data
            const response = await fetch("http://localhost:8000/restart-simulation/");
            if (!response.ok) throw new Error(`Init failed: ${response.status}`);

            const data = await response.json();

            if (data.status === "success") {
                console.log("✅ Simulation initialized successfully");

                // Open simulation window
                const simWindow = window.open(
                    "/simulation",
                    "_blank",
                    "width=1000,height=600,resizable=yes,scrollbars=yes"
                );

                if (!simWindow) {
                    console.error("❌ Failed to open simulation window. Check popup blockers.");
                    window.alert("❌ Failed to open simulation window. Check popup blockers.");
                }
            } else {
                console.error("❌ Backend init failed:", data.message);
                window.alert(`❌ Backend error: ${data.message}`);
            }
        } catch (error) {
            console.error("🚨 Simulation start failed:", error);
            window.alert("🚨 Error initializing simulation.");
        }

        setLoading(false);
    };

    return (
        <button
            onClick={handleClick}
            disabled={loading}
            style={{
                backgroundColor: loading ? '#888' : '#1f78b4',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: loading ? 'not-allowed' : 'pointer',
                margin: '5px',
                opacity: loading ? 0.6 : 1
            }}
        >
            {loading ? "⏳ Starting..." : "🚀 Start Simulation"}
        </button>
    );
}

export default StartSimulationButton;
