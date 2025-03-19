import React, { useState } from "react";

function TrainRegressionModelButton({ onRegressionComplete }) {
    const [showModal, setShowModal] = useState(false);
    const [dropPriceColumns, setDropPriceColumns] = useState(true);
    const [applyFilter, setApplyFilter] = useState(true);
    const [filterThreshold, setFilterThreshold] = useState(4);
    const [error, setError] = useState(null);

    const handleOpenModal = () => setShowModal(true);

    const handleConfirmTraining = async () => {
        setShowModal(false);

        try {
            const response = await fetch("http://127.0.0.1:8000/train-regression-model/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    drop_price_columns: dropPriceColumns,
                    apply_filter: applyFilter,
                    filter_threshold: filterThreshold
                })
            });

            const result = await response.json();
            if (result.status === "success") {
                console.log("Regression Training Completed:", result);
                setError(null);

                // ✅ Pass results to the parent component
                onRegressionComplete(result);

            } else {
                setError(result.message);
            }
        } catch (err) {
            setError("⚠️ Failed to connect to backend.");
            console.error("Fetch error:", err);
        }
    };

    return (
        <div>
            <button style={buttonStyle} onClick={handleOpenModal}>
                📊 Train Regression Model
            </button>

            {showModal && (
                <div style={modalOverlayStyle}>
                    <div style={modalStyle}>
                        <h3>Training Options</h3>

                        {/* ✅ Checkbox for including price columns */}
                        <label style={{ display: 'block', margin: '10px 0' }}>
                            <input
                                type="checkbox"
                                checked={!dropPriceColumns}
                                onChange={(e) => setDropPriceColumns(!e.target.checked)}
                            />
                            <span style={{ marginLeft: '8px' }}>Include Prices</span>
                        </label>

                        {/* ✅ Checkbox to apply filtering */}
                        <label style={{ display: 'block', margin: '10px 0' }}>
                            <input
                                type="checkbox"
                                checked={applyFilter}
                                onChange={(e) => setApplyFilter(e.target.checked)}
                            />
                            <span style={{ marginLeft: '8px' }}>Apply Filtering</span>
                        </label>

                        {/* ✅ Input for user-defined filter threshold */}
                        <label style={{ display: 'block', margin: '10px 0' }}>
                            <span>Filter Predictions Above (Absolute Error):</span>
                            <input
                                type="number"
                                value={filterThreshold}
                                onChange={(e) => setFilterThreshold(parseFloat(e.target.value))}
                                style={inputStyle}
                                disabled={!applyFilter}
                            />
                        </label>

                        <div style={{ marginTop: '15px' }}>
                            <button style={modalButtonStyle} onClick={handleConfirmTraining}>Confirm ✅</button>
                            <button style={cancelButtonStyle} onClick={() => setShowModal(false)}>Cancel ❌</button>
                        </div>
                    </div>
                </div>
            )}

            {error && <p style={{ color: "red" }}>⚠️ {error}</p>}
        </div>
    );
}

// ✅ Styles
const buttonStyle = {
    padding: '10px 20px',
    backgroundColor: '#6f42c1',
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

const cancelButtonStyle = {
    padding: '10px 20px',
    margin: '5px',
    backgroundColor: '#dc3545',
    color: 'white',
    borderRadius: '5px',
    cursor: 'pointer'
};

const inputStyle = {
    marginLeft: "8px",
    padding: "5px",
    borderRadius: "5px",
    border: "1px solid #ccc",
    width: "60px",
};

export default TrainRegressionModelButton;
