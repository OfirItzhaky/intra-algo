import React from "react"; // ✅ Ensure React is imported

function DataSummaryBar({
    trainingSummary,
    simulatingSummary,
    labelSummary,
    newFeaturesCount,
    regressionMetrics,
    classifierResults
}) {
    console.log("🔍 Debugging Classifier Results:", classifierResults); // ✅ Debug Log

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

            {/* ✅ Regression Metrics */}
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

            {/* ✅ Classifier Results Table */}
            {classifierResults && (
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #facc15' }}>
                    <strong>🤖 Classifier Performance:</strong>
                    <table style={tableStyle}>
                        <thead>
                            <tr>
                                <th>Classifier</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(classifierResults).map(([model, results]) => (
                                <React.Fragment key={model}>
                                    {/* ✅ Overall Results */}
                                    <tr>
                                        <td>{model} (Overall)</td>
                                        <td>{results?.precision_1 !== undefined ? (results.precision_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>{results?.recall_1 !== undefined ? (results.recall_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>{results?.f1_1 !== undefined ? (results.f1_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>{results?.accuracy !== undefined ? (results.accuracy * 100).toFixed(1) + "%" : "N/A"}</td>
                                    </tr>

                                    {/* ✅ Label 1 Results (Bold & Different Color) */}
                                    <tr style={labelOneStyle}>
                                        <td>↳ {model} (Label 1)</td>
                                        <td>{results?.precision_1 !== undefined ? (results.precision_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>{results?.recall_1 !== undefined ? (results.recall_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>{results?.f1_1 !== undefined ? (results.f1_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        <td>N/A</td> {/* Accuracy is not relevant per label */}
                                    </tr>
                                </React.Fragment>
                            ))}
                        </tbody>
                    </table>
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

// ✅ Style for Classifier Results Table
const tableStyle = {
    width: '100%',
    marginTop: '10px',
    borderCollapse: 'collapse',
    textAlign: 'left',
    backgroundColor: '#1e293b',
    color: '#facc15'
};

// ✅ Style for Label 1 Rows
const labelOneStyle = {
    color: "#4ade80", // Brighter yellow to stand out
    fontWeight: "bold",
    fontSize: "1.1em"
};

export default DataSummaryBar;
