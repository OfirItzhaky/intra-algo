import React from 'react';

function RestartSimulationButton() {
    const handleClick = () => {
        console.log('Restart Simulation Clicked!');
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
