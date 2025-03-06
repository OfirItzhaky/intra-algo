import React from 'react';

function DataSummaryBar({ trainingSummary, simulatingSummary }) {
    return (
        <div style={statusBarStyle}>
            <h4>📊 Current Data Summary</h4>
            <p><strong>Training Data:</strong> {trainingSummary ? `${trainingSummary.first_date} to ${trainingSummary.last_date}` : "Not Loaded"}</p>
            <p><strong>Simulation Data:</strong> {simulatingSummary ? `${simulatingSummary.first_date} to ${simulatingSummary.last_date}` : "Not Loaded"}</p>
        </div>
    );
}

const statusBarStyle = {
    position: 'fixed',
    top: '10px',
    right: '10px',
    backgroundColor: '#1e293b',
    color: '#a7f3d0',
    padding: '10px',
    borderRadius: '8px',
    fontFamily: 'monospace',
    zIndex: 1000
};

export default DataSummaryBar;
