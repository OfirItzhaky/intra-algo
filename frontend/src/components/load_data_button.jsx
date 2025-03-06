import React, { useState, useRef } from 'react';

function LoadDataButton({ onSummaryLoaded }) {
    const [showModal, setShowModal] = useState(false);
    const [dataType, setDataType] = useState("training");
    const [symbol, setSymbol] = useState("MES");
    const fileInputRef = useRef(null);
    const [error, setError] = useState(null);

    const handleOpenModal = () => setShowModal(true);

    const handleConfirmDataType = () => {
        setShowModal(false);
        fileInputRef.current.click();
    };

    const handleFileSelected = (event) => {
        const file = event.target.files[0];
        if (!file) {
            setError("No file selected.");
            return;
        }

        const filePath = file.name;
        const url = `http://127.0.0.1:8000/load-data/?file_path=${filePath}&data_type=${dataType}&symbol=${symbol}`;

        fetch(url)
            .then(res => res.json())
            .then(result => {
                if (result.status === "success") {
                    const loadedSummary = {
                        ...result.summary,   // Include backend summary (dates/times)
                        dataType              // training or simulating
                    };

                    onSummaryLoaded(loadedSummary);  // Send to app.jsx

                    setError(null);
                } else {
                    setError(result.message);
                    onSummaryLoaded({ dataType, error: result.message });  // Pass error to summary bar too
                }
            })
            .catch(() => {
                setError("Failed to reach backend.");
                onSummaryLoaded({ dataType, error: "Failed to reach backend." });
            });

    };

    return (
        <div>
            <input
                type="file"
                accept=".csv,.txt"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileSelected}
            />

            <button style={buttonStyle} onClick={handleOpenModal}>
                📂 Load Data
            </button>

            {showModal && (
                <div style={modalOverlayStyle}>
                    <div style={modalStyle}>
                        <h3>Select Data Type & Symbol</h3>

                        <label>
                            <input type="radio" checked={dataType === "training"} onChange={() => setDataType("training")} />
                            Training Data
                        </label>
                        <label style={{ marginLeft: '15px' }}>
                            <input type="radio" checked={dataType === "simulating"} onChange={() => setDataType("simulating")} />
                            Simulation Data
                        </label>

                        <div style={{ marginTop: '10px' }}>
                            <label>Symbol:</label>
                            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
                                <option value="MES">MES - Micro E-mini S&P</option>
                                <option value="NQ">NQ - Nasdaq 100</option>
                            </select>
                        </div>

                        <div style={{ marginTop: '15px' }}>
                            <button style={modalButtonStyle} onClick={handleConfirmDataType}>Next ➡️</button>
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
    backgroundColor: '#10b981',
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

export default LoadDataButton;
