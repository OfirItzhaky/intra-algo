import React from 'react';

function DataSummaryBar({ trainingSummary, simulatingSummary }) {
    return (
        <div style={barStyle}>
            <h4 style={{ marginBottom: '10px', textDecoration: 'underline' }}>📊 Current Data Summary</h4>

            {/* Training Summary */}
            <div style={summaryItemStyle}>
                <strong>Training Data:</strong> {trainingSummary
                    ? `${trainingSummary.first_date} ${trainingSummary.first_time} ➡️ ${trainingSummary.last_date} ${trainingSummary.last_time}`
                    : "Not Loaded"}
                {trainingSummary?.error && (
                    <p style={warningStyle}>⚠️ {trainingSummary.error}</p>
                )}
            </div>

            {/* Simulation Summary */}
            <div style={summaryItemStyle}>
                <strong>Simulation Data:</strong> {simulatingSummary
                    ? `${simulatingSummary.first_date} ${simulatingSummary.first_time} ➡️ ${simulatingSummary.last_date} ${simulatingSummary.last_time}`
                    : "Not Loaded"}
                {simulatingSummary?.error && (
                    <p style={warningStyle}>⚠️ {simulatingSummary.error}</p>
                )}
            </div>
        </div>
    );
}

// Styles inside the file (no external CSS needed)
const barStyle = {
    position: 'absolute',
    top: '15px',
    right: '15px',
    backgroundColor: '#1e293b',
    padding: '12px 16px',
    borderRadius: '8px',
    color: '#a7f3d0',
    fontFamily: 'monospace',
    boxShadow: '0 2px 10px rgba(0,0,0,0.5)',
    zIndex: 100
};

const summaryItemStyle = {
    marginBottom: '8px',
    lineHeight: '1.4'
};

const warningStyle = {
    color: '#fbbf24',
    fontSize: '12px',
    marginTop: '3px'
};

export default DataSummaryBar;
