import React from 'react';

function ValidateDataButton() {
    const handleClick = () => {
        console.log('Validate Data Clicked!');
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#007bff',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            ✅ Validate Data
        </button>
    );
}

export default ValidateDataButton;
