import React from 'react';

function TrainClassifiersButton() {
    const handleClick = () => {
        console.log('Train Classifiers Clicked!');
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#dc3545',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            📑 Train Classifiers
        </button>
    );
}

export default TrainClassifiersButton;
