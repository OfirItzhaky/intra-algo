import React, { useState } from 'react';
import './index.css';
import LoadDataButton from './components/load_data_button';
import ValidateDataButton from './components/validate_data_button';
import GenerateFeaturesButton from './components/generate_features_button';
import StartSimulationButton from './components/start_simulation_button';
import TrainRegressionModelButton from './components/train_regression_model_button';
import TrainClassifiersButton from './components/train_classifiers_button';
import RestartSimulationButton from './components/restart_simulation_button';
import DataSummaryBar from './components/data_summary_bar';  // Make sure this file exists

function App() {
    const [trainingSummary, setTrainingSummary] = useState(null);
    const [simulatingSummary, setSimulatingSummary] = useState(null);

    const handleSummaryLoaded = (summary) => {
        if (summary?.dataType === 'training') {
            setTrainingSummary(summary);
        } else if (summary?.dataType === 'simulating') {
            setSimulatingSummary(summary);
        }
    };

    return (
        <div className="app-container">
            <div className="content-box">
                <h1 className="app-title">Intra Algo Trading Platform</h1>
                <p className="app-subtitle">Intra Algo Simulator - Version 1.0</p>

                {/* Status Bar for Loaded Data (always visible) */}
                <DataSummaryBar
                    trainingSummary={trainingSummary}
                    simulatingSummary={simulatingSummary}
                />

                {/* Buttons Area */}
                <div className="button-group">
                    <LoadDataButton onSummaryLoaded={handleSummaryLoaded} />
                    <ValidateDataButton />
                    <GenerateFeaturesButton />
                    <TrainRegressionModelButton />
                    <TrainClassifiersButton />
                    <StartSimulationButton />
                    <RestartSimulationButton />
                </div>
            </div>
        </div>
    );
}

export default App;
