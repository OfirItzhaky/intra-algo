function DataSummaryBar({ trainingSummary, simulatingSummary, labelSummary, newFeaturesCount, regressionMetrics }) {
    console.log("🔍 Debugging Regression Metrics:", regressionMetrics); // ✅ Debug Log

    return (
        <div style={barStyle}>
            <h4 style={{ marginBottom: '10px', textDecoration: 'underline' }}>📊 Current Data Summary</h4>

            <div style={summaryItemStyle}>
                <strong>Training Data:</strong> {trainingSummary
                    ? `${trainingSummary.first_date} ${trainingSummary.first_time} ➡️ ${trainingSummary.last_date} ${trainingSummary.last_time}`
                    : "Not Loaded"}
            </div>

            <div style={summaryItemStyle}>
                <strong>Simulation Data:</strong> {simulatingSummary
                    ? `${simulatingSummary.first_date} ${simulatingSummary.first_time} ➡️ ${simulatingSummary.last_date} ${simulatingSummary.last_time}`
                    : "Not Loaded"}
            </div>

            <div style={summaryItemStyle}>
                <strong>Features Created:</strong> {newFeaturesCount !== null
                    ? `${newFeaturesCount} features added`
                    : "Not Generated"}
            </div>

            <div style={summaryItemStyle}>
                <strong>Labels Generated:</strong> {labelSummary
                    ? `${labelSummary.label_type} (${labelSummary.rows_labeled} rows)`
                    : "Not Generated"}
            </div>

            {/* ✅ Regression Metrics (NEW) */}
            {regressionMetrics && (
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #4ade80' }}>
                    <strong>📉 Regression Performance:</strong>
                    <div style={summaryItemStyle}>
                        <strong>MSE:</strong> {regressionMetrics.mse_filtered}
                    </div>
                    <div style={summaryItemStyle}>
                        <strong>R² Score:</strong> {regressionMetrics.r2_filtered}
                    </div>
                </div>
            )}
        </div>
    );
}

// ✅ Styles remain unchanged
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

export default DataSummaryBar;
