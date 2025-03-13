import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
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
import VisualizeClassifiersButton from './components/visualize_classifiers_button';
import SimulationScreen from './components/simulation_screen'; // ✅ New Simulation Component

function App() {
    const [trainingSummary, setTrainingSummary] = useState(null);
    const [simulatingSummary, setSimulatingSummary] = useState(null);
    const [labelSummary, setLabelSummary] = useState(null);
    const [featuresCount, setFeaturesCount] = useState(null);
    const [regressionMetrics, setRegressionMetrics] = useState(null);
    const [classifierResults, setClassifierResults] = useState(null);
    const [classifierVisualization, setClassifierVisualization] = useState(null);

    const handleSummaryLoaded = (summary) => {
        if (summary?.dataType === 'training') {
            setTrainingSummary(summary);
        } else if (summary?.dataType === 'simulating') {
            setSimulatingSummary(summary);
        }
    };

    return (
        <Router>
            <Routes>
                <Route path="/" element={
                    <div className="app-container">
                        <div className="content-box">
                            <h1 className="app-title">Intra Algo Trading Platform</h1>
                            <p className="app-subtitle">Intra Algo Simulator - Version 1.0</p>

                            <DataSummaryBar
                                trainingSummary={trainingSummary}
                                simulatingSummary={simulatingSummary}
                                labelSummary={labelSummary}
                                newFeaturesCount={featuresCount}
                                regressionMetrics={regressionMetrics}
                                classifierResults={classifierResults}
                                classifierVisualization={classifierVisualization}
                            />

                            <div className="button-group">
                                <LoadDataButton onSummaryLoaded={handleSummaryLoaded} />  {/* ✅ Restore this */}
                                <ValidateDataButton />
                                <GenerateFeaturesButton onFeaturesGenerated={setFeaturesCount} />
                                <GenerateLabelsButton onLabelsGenerated={setLabelSummary} />
                                <TrainRegressionModelButton onRegressionComplete={setRegressionMetrics} />
                                <TrainClassifiersButton onClassificationComplete={setClassifierResults} />
                                <VisualizeClassifiersButton onVisualizationComplete={setClassifierVisualization} />
                                <StartSimulationButton />
                                <RestartSimulationButton />
                            </div>
                        </div>
                    </div>
                } />
                <Route path="/simulation" element={<SimulationScreen />} />
            </Routes>
        </Router>
    );
}


export default App;
