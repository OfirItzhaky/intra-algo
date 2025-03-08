import React, { useState } from 'react';
import './index.css';
import LoadDataButton from './components/load_data_button';
import ValidateDataButton from './components/validate_data_button';
import GenerateFeaturesButton from './components/generate_features_button';
import GenerateLabelsButton from './components/generate_labels_button';
import StartSimulationButton from './components/start_simulation_button';
import TrainRegressionModelButton from './components/train_regression_model_button';
import TrainClassifiersButton from './components/train_classifiers_button';
import RestartSimulationButton from './components/restart_simulation_button';
import DataSummaryBar from './components/data_summary_bar';

function App() {
    const [trainingSummary, setTrainingSummary] = useState(null);
    const [simulatingSummary, setSimulatingSummary] = useState(null);
    const [labelSummary, setLabelSummary] = useState(null);
    const [featuresCount, setFeaturesCount] = useState(null); // ✅ Track features count

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

                {/* ✅ Updated Summary Bar */}
                <DataSummaryBar
                    trainingSummary={trainingSummary}
                    simulatingSummary={simulatingSummary}
                    labelSummary={labelSummary}
                    newFeaturesCount={featuresCount} // ✅ Pass features count to the status bar
                />

                {/* ✅ Buttons Area */}
                <div className="button-group">
                    <LoadDataButton onSummaryLoaded={handleSummaryLoaded} />
                    <ValidateDataButton />
                    <GenerateFeaturesButton onFeaturesGenerated={setFeaturesCount} /> {/* ✅ Pass the update function */}
                    <GenerateLabelsButton onLabelsGenerated={setLabelSummary} />
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
