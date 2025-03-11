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
    const [featuresCount, setFeaturesCount] = useState(null);
    const [regressionMetrics, setRegressionMetrics] = useState(null); // ✅ Store regression results
    const [classifierResults, setClassifierResults] = useState(null); // ✅ Store classifier results

    const handleSummaryLoaded = (summary) => {
        if (summary?.dataType === 'training') {
            setTrainingSummary(summary);
        } else if (summary?.dataType === 'simulating') {
            setSimulatingSummary(summary);
        }
    };

    // ✅ New function to handle classifier results correctly
    const handleClassificationComplete = (results) => {
        console.log("✅ Updating Classifier Results:", results); // Debugging API Response

        if (results && results.RandomForest && results.LightGBM && results.XGBoost) {
            console.log("✅ Valid classifier results received. Updating state.");

            setClassifierResults({
                RandomForest: results.RandomForest,
                LightGBM: results.LightGBM,
                XGBoost: results.XGBoost
            });

            console.log("✅ State Updated! New classifierResults:", {
                RandomForest: results.RandomForest,
                LightGBM: results.LightGBM,
                XGBoost: results.XGBoost
            });
        } else {
            console.warn("⚠️ Received invalid classifier results:", results);
        }
    };


    return (
        <div className="app-container">
            <div className="content-box">
                <h1 className="app-title">Intra Algo Trading Platform</h1>
                <p className="app-subtitle">Intra Algo Simulator - Version 1.0</p>

                {/* ✅ Updated Summary Bar with Regression and Classification Metrics */}
                <DataSummaryBar
                    trainingSummary={trainingSummary}
                    simulatingSummary={simulatingSummary}
                    labelSummary={labelSummary}
                    newFeaturesCount={featuresCount}
                    regressionMetrics={regressionMetrics} // ✅ Pass regression results
                    classifierResults={classifierResults} // ✅ Pass classifier results
                />

                {/* ✅ Buttons Area */}
                <div className="button-group">
                    <LoadDataButton onSummaryLoaded={handleSummaryLoaded} />
                    <ValidateDataButton />
                    <GenerateFeaturesButton onFeaturesGenerated={setFeaturesCount} />
                    <GenerateLabelsButton onLabelsGenerated={setLabelSummary} />
                    <TrainRegressionModelButton onRegressionComplete={setRegressionMetrics} />
                    <TrainClassifiersButton onClassificationComplete={handleClassificationComplete} /> {/* ✅ Fix: Use new function */}
                    <StartSimulationButton />
                    <RestartSimulationButton />
                </div>
            </div>
        </div>
    );
}

export default App;
