import React from 'react';

function StartSimulationButton() {
    const handleClick = () => {
        console.log('Start Simulation Clicked!');
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#ff9900',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            🚀 Start Simulation
        </button>
    );
}

export default StartSimulationButton;
