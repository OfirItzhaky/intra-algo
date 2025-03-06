import React from 'react';

function TrainModelButton() {
    const handleClick = () => {
        console.log('Train Model Clicked!');
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#6f42c1',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            📊 Train Model
        </button>
    );
}

export default TrainModelButton;
