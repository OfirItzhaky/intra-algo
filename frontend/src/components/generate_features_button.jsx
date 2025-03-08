import React, { useState } from 'react';

function GenerateFeaturesButton({ onFeaturesGenerated }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleClick = () => {
        setLoading(true);
        setError(null);

        fetch("http://127.0.0.1:8000/generate-features/", { method: "POST" })
            .then(res => res.json())
            .then(result => {
                setLoading(false);
                if (result.status === "success") {
                    onFeaturesGenerated(result.new_features_count);
                } else {
                    setError(result.message);
                }
            })
            .catch(() => {
                setLoading(false);
                setError("Failed to connect to backend.");
            });
    };

    return (
        <div>
            <button style={buttonStyle} onClick={handleClick} disabled={loading}>
                {loading ? "Generating..." : "🛠️ Generate Features"}
            </button>
            {error && <p style={{ color: 'red' }}>⚠️ {error}</p>}
        </div>
    );
}

const buttonStyle = {
    padding: '10px 20px',
    backgroundColor: '#f59e0b',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '16px',
    marginTop: '10px'
};

export default GenerateFeaturesButton;
