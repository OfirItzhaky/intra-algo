import React from 'react';

function RestartSimulationButton({ onRestart }) {
    const handleClick = async () => {
        console.log('🔄 Restart Simulation Clicked!');

        try {
            const response = await fetch("http://localhost:8000/restart-simulation/");
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

            const data = await response.json();

            if (data.status === "success") {
                console.log("✅ Simulation restarted successfully!");
                if (typeof onRestart === "function") {
                    onRestart();  // ✅ Notify parent to refresh UI
                }
            } else {
                console.error("❌ Error restarting simulation:", data.message);
            }
        } catch (error) {
            console.error("🚨 Failed to restart simulation:", error);
        }
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#17a2b8',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            🔄 Restart Simulation
        </button>
    );
}

export default RestartSimulationButton;
