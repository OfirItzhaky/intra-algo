import React from 'react';

function GenerateFeaturesButton() {
    const handleClick = () => {
        console.log('Generate Features Clicked!');
    };

    return (
        <button
            onClick={handleClick}
            style={{
                backgroundColor: '#28a745',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                margin: '5px'
            }}
        >
            ⚙️ Generate Features
        </button>
    );
}

export default GenerateFeaturesButton;
