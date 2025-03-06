import React from 'react';

function LoadDataButton() {
    return (
        <button style={buttonStyle}>
            📂 Load Data
        </button>
    );
}

const buttonStyle = {
    padding: '10px 20px',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px'
};

export default LoadDataButton;
