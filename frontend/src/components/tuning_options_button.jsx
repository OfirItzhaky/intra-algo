import React, { useState, useEffect } from 'react';

function GenerateClassifierLabelButton({ onLabelMethodChange, onLabelGenerated }) {
    const [loading, setLoading] = useState(false);
    const [labelMethod, setLabelMethod] = useState('add_good_bar_label');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    // Label method options
    const labelMethods = [
        { value: 'add_good_bar_label', label: 'Standard Label (Good Bar Prediction)' },
        { value: 'long_good_bar_label_all', label: 'Long Good Bar (All Bars)' },
        { value: 'long_good_bar_label_bullish_only', label: 'Long Good Bar (Bullish Only)' }
    ];

    // When labelMethod changes, update the parent component
    useEffect(() => {
        const selectedMethod = labelMethods.find(m => m.value === labelMethod);
        if (selectedMethod) {
            onLabelMethodChange(labelMethod, selectedMethod.label);
        }
    }, [labelMethod, onLabelMethodChange]);

    const handleClick = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        
        try {
            const response = await fetch(`http://127.0.0.1:8000/generate-classifier-labels/?label_method=${labelMethod}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log("🏷️ Label generation successful:", data);
                setResult(data);
                // Notify parent component
                if (onLabelGenerated) {
                    onLabelGenerated(data);
                }
            } else {
                console.error("⚠️ Label generation failed:", data.message);
                setError(data.message);
            }
        } catch (err) {
            console.error("🚨 API request failed:", err);
            setError("Failed to connect to the server.");
        }

        setLoading(false);
    };

    return (
        <div className="label-method-container">
            <div className="label-method-row">
                <label htmlFor="labelMethodSelect" className="label-method-label">
                    Label Method:
                </label>
                <select
                    id="labelMethodSelect"
                    value={labelMethod}
                    onChange={(e) => setLabelMethod(e.target.value)}
                    className="label-method-select"
                    disabled={loading}
                >
                    {labelMethods.map(method => (
                        <option key={method.value} value={method.value}>
                            {method.label}
                        </option>
                    ))}
                </select>
            </div>

            <button
                onClick={handleClick}
                disabled={loading}
                className="label-method-button"
            >
                {loading ? '⏳ Generating...' : '🏷️ Generate Classifier Label'}
            </button>

            {error && (
                <div className="label-method-error">
                    ⚠️ {error}
                </div>
            )}

            {result && (
                <div className="label-method-result">
                    <div>✅ {result.message}</div>
                    <div>📊 Total labels: {result.total_labels}</div>
                    <div>🔍 Positive labels: {result.positive_labels} ({result.positive_percentage}%)</div>
                </div>
            )}
        </div>
    );
}

export default GenerateClassifierLabelButton; 