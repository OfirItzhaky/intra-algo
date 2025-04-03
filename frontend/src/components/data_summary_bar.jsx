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

    // Format the label method nicely
    const formatLabelMethod = (method) => {
        if (!method) return "";
        
        // Convert from snake_case to readable format
        let formattedMethod = method
            .replace('add_', '')
            .replace('long_', '')
            .replace('_goal_c', '')  // Remove goal_c suffix
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
            
        // Add "Next Bar" prefix for goal_c methods
        if (method.includes('goal_c')) {
            formattedMethod = 'Next Bar ' + formattedMethod;
        }
        
        return formattedMethod;
    };

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

                {/* ✅ Display Warnings */}
                {simulatingSummary?.missing_data_warning && (
                    <div style={warningStyle}>⚠️ {simulatingSummary.missing_data_warning}</div>
                )}
                {simulatingSummary?.insufficient_simulation_warning && (
                    <div style={errorStyle}>❌ {simulatingSummary.insufficient_simulation_warning}</div>
                )}
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
                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
                        <strong>🤖 Classifier Performance:</strong>
                        <span style={{...labelMethodStyle, fontSize: '1em', padding: '3px 10px'}}>
                            Label: <span style={{fontWeight: 'bold'}}>
                                {classifierResults.labelDisplayName || 
                                 (classifierResults.labelMethod ? formatLabelMethod(classifierResults.labelMethod) : "Good Bar Label")}
                            </span>
                        </span>
                    </div>
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
                            {Object.entries(classifierResults).map(([model, results]) => {
                                // Skip the cv_results entry and other non-model properties
                                if (model === 'cv_results' || model === 'labelMethod' || model === 'targetColumn') return null;
                                
                                return (
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
                            )})}
                        </tbody>
                    </table>
                    
                    {/* ✅ Cross-Validation Results Table */}
                    {classifierResults.cv_results && (
                        <div style={{ marginTop: '15px', paddingTop: '10px', borderTop: '1px dashed #a78bfa' }}>
                            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
                                <strong>🔄 Cross-Validation Performance:</strong>
                                <span style={{...cvLabelMethodStyle, fontSize: '1em', padding: '3px 10px'}}>
                                    Label: <span style={{fontWeight: 'bold'}}>
                                        {classifierResults.labelDisplayName || 
                                         (classifierResults.labelMethod ? formatLabelMethod(classifierResults.labelMethod) : "Good Bar Label")}
                                    </span>
                                </span>
                            </div>
                            <table style={cvTableStyle}>
                                <thead>
                                    <tr>
                                        <th style={{...cvHeaderStyle, textAlign: 'left'}}>Classifier</th>
                                        <th style={cvHeaderStyle}>F1 (Label 1)</th>
                                        <th style={cvHeaderStyle}>Precision (Label 1)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(classifierResults.cv_results).map(([model, results]) => (
                                        <tr key={model} style={model === 'XGBoost' ? cvBestResultStyle : null}>
                                            <td style={cvCellStyle}>{model}</td>
                                            <td style={cvValueStyle}>{results?.F1_Label_1 !== undefined ? (results.F1_Label_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                            <td style={cvValueStyle}>{results?.Precision_Label_1 !== undefined ? (results.Precision_Label_1 * 100).toFixed(1) + "%" : "N/A"}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
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
    maxWidth: '450px', // ✅ Limit width to prevent overflow
    backgroundColor: '#1e293b',
    padding: '12px 16px',
    borderRadius: '8px',
    color: '#a7f3d0',
    fontFamily: 'monospace',
    boxShadow: '0 2px 10px rgba(0,0,0,0.5)',
    zIndex: 100,
    overflow: 'auto', // ✅ Allow scrolling if content overflows
    wordWrap: 'break-word' // ✅ Prevent text overflow issues
};


const summaryItemStyle = {
    marginBottom: '8px',
    lineHeight: '1.4'
};

// ✅ Warning & Error Styles
const warningStyle = {
    color: "#facc15", // Yellow warning
    fontWeight: "bold",
    marginTop: "5px",
    padding: "5px", // ✅ Add padding to improve spacing
    borderRadius: "5px",
    backgroundColor: "rgba(250, 201, 21, 0.15)" // ✅ Light yellow background
};

const errorStyle = {
    color: "#f87171", // Red error
    fontWeight: "bold",
    marginTop: "5px",
    padding: "5px", // ✅ Add padding
    borderRadius: "5px",
    backgroundColor: "rgba(248, 113, 113, 0.15)" // ✅ Light red background
};

// ✅ Style for label method display
const labelMethodStyle = {
    color: "#fb923c", // Changed to orange instead of yellow
    fontSize: '0.9em',
    padding: "2px 8px",
    borderRadius: "4px",
    backgroundColor: "rgba(251, 146, 60, 0.15)",  // Orange background
    border: "1px solid rgba(251, 146, 60, 0.3)",
    fontWeight: "500"
};

// ✅ Style for CV label method display
const cvLabelMethodStyle = {
    color: "#c084fc", // Changed to brighter purple
    fontSize: '0.9em',
    padding: "2px 8px",
    borderRadius: "4px",
    backgroundColor: "rgba(192, 132, 252, 0.15)", // Bright purple background
    border: "1px solid rgba(192, 132, 252, 0.3)",
    fontWeight: "500"
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
    color: "#4ade80",
    fontWeight: "bold",
    fontSize: "1.1em"
};

// ✅ Style for CV Table
const cvTableStyle = {
    width: '100%',
    marginTop: '10px',
    borderCollapse: 'collapse',
    textAlign: 'center',
    backgroundColor: '#1e293b',
    color: '#a78bfa', // Purple color for CV table
    tableLayout: 'fixed' // Force equal column widths
};

// ✅ Style for CV headers to keep them on one line
const cvHeaderStyle = {
    whiteSpace: 'nowrap',
    padding: '0 10px',
    textAlign: 'center',
    borderBottom: '1px solid #4c347a'
};

// ✅ Style for CV cell (first column)
const cvCellStyle = {
    textAlign: 'left',
    padding: '3px 10px'
};

// ✅ Style for CV value cells (numeric data)
const cvValueStyle = {
    textAlign: 'center',
    padding: '3px 10px'
};

// ✅ Style for best CV model
const cvBestResultStyle = {
    color: "#c4b5fd",
    fontWeight: "bold",
    fontSize: "1.1em"
};

export default DataSummaryBar;
