import React, { useState } from 'react';

function StartSimulationButton() {
    const [loading, setLoading] = useState(false);

    const handleClick = () => {
        if (loading) return; // Prevent multiple clicks

        console.log('🚀 Opening Simulation Window...');
        setLoading(true);

        // Open a new simulation window
        const simWindow = window.open(
            "/simulation", // We’ll define this route in React
            "_blank",
            "width=1000,height=600,resizable=yes,scrollbars=yes"
        );

        if (!simWindow) {
            console.error("❌ Failed to open simulation window. Check popup blockers.");
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
