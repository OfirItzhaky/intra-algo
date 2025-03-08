import React, { useState } from 'react';

function GenerateLabelsButton({ onLabelsGenerated }) {
    const [showModal, setShowModal] = useState(false);
    const [selectedLabel, setSelectedLabel] = useState('next_high');
    const [error, setError] = useState(null);

    const handleOpenModal = () => setShowModal(true);

    const handleConfirmLabel = () => {
        setShowModal(false);

        fetch(`http://127.0.0.1:8000/generate-labels/?label_type=${selectedLabel}`)
            .then(response => response.json())
            .then(result => {
                if (result.status === "success") {
                    onLabelsGenerated(result.summary); // ✅ Update App.jsx state
                    setError(null);
                } else {
                    setError(result.message);
                }
            })
            .catch(() => setError("⚠️ Failed to reach backend."));
    };

    return (
        <div>
            <button style={buttonStyle} onClick={handleOpenModal}>
                🏷️ Generate Labels
            </button>

            {showModal && (
                <div style={modalOverlayStyle}>
                    <div style={modalStyle}>
                        <h3>Select Label Type</h3>
                        <label>
                            <input
                                type="radio"
                                name="labelType"
                                value="next_high"
                                checked={selectedLabel === "next_high"}
                                onChange={() => setSelectedLabel("next_high")}
                            />
                            Next High (Elastic Net)
                        </label>
                        <br />
                        <label>
                            <input
                                type="radio"
                                name="labelType"
                                value="good_bar"
                                checked={selectedLabel === "good_bar"}
                                onChange={() => setSelectedLabel("good_bar")}
                            />
                            Good Bar (Classifier)
                        </label>

                        <div style={{ marginTop: '15px' }}>
                            <button style={modalButtonStyle} onClick={handleConfirmLabel}>Generate ➡️</button>
                            <button style={modalButtonStyle} onClick={() => setShowModal(false)}>Cancel ✖️</button>
                        </div>
                    </div>
                </div>
            )}

            {error && <p style={{ color: 'red' }}>⚠️ {error}</p>}
        </div>
    );
}



// Styles
const buttonStyle = {
    padding: '10px 20px',
    backgroundColor: '#f59e0b',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '16px'
};

const modalOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
};

const modalStyle = {
    backgroundColor: '#1e293b',
    padding: '20px',
    borderRadius: '12px',
    color: '#e2e8f0',
    width: '350px',
    textAlign: 'center'
};

const modalButtonStyle = {
    padding: '10px 20px',
    margin: '5px',
    backgroundColor: '#10b981',
    color: 'white',
    borderRadius: '5px',
    cursor: 'pointer'
};

export default GenerateLabelsButton;
