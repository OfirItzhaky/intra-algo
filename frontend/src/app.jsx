import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './index.css';
import LoadDataButton from './components/load_data_button';
import ValidateDataButton from './components/validate_data_button';
import GenerateFeaturesButton from './components/generate_features_button';
import GenerateLabelsButton from './components/generate_labels_button';
import StartSimulationButton from './components/start_simulation_button';
import TrainRegressionModelButton from './components/train_regression_model_button';
import VisualizeRegressionButton from './components/visualize_regression_button'; // ✅ New button
import GenerateClassifierLabelButton from './components/tuning_options_button'; // ✅ New classifier label button
import TrainClassifiersButton from './components/train_classifiers_button';
import RestartSimulationButton from './components/restart_simulation_button';
import DataSummaryBar from './components/data_summary_bar';
import VisualizeClassifiersButton from './components/visualize_classifiers_button';
import SimulationScreen from './components/simulation_screen'; // ✅ New Simulation Component
import Simulation2 from './components/simulation2'; // ✅ New Simulation Component
import Simulation3 from './components/simulation3'; // ✅ New Simulation Component
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
            if (summary.fixed_simulation_df) {  // ✅ Ensure we use the fixed version if available
                const fixedFirstRow = summary.fixed_simulation_df[0];  // ✅ Extract first row from fixed data
                const fixedLastRow = summary.fixed_simulation_df[summary.fixed_simulation_df.length - 1]; // ✅ Last row

                setSimulatingSummary({
                    ...summary,
                    first_date: fixedFirstRow.Date,   // ✅ Correctly extract first date
                    first_time: fixedFirstRow.Time,
                    last_date: fixedLastRow.Date,     // ✅ Correctly extract last date
                    last_time: fixedLastRow.Time
                });
            } else {
                setSimulatingSummary(summary);  // ✅ Fallback if no fixed data
            }
        }
    };

    const handleSimulationRestart = async () => {
        console.log("🔄 Restarting simulation... (Summary remains unchanged)");

        try {
            const response = await fetch("http://localhost:8000/restart-simulation");
            const data = await response.json();

            if (data.status === "success") {
                console.log("✅ Simulation restarted successfully!");
                // ✅ Optional: Add a toast/pop-up to confirm restart if needed
            } else {
                console.error("❌ Error restarting simulation:", data.message);
            }
        } catch (error) {
            console.error("🚨 Failed to restart simulation:", error);
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
                                simulatingSummary={simulatingSummary}  // ✅ Summary remains unchanged on restart
                                labelSummary={labelSummary}
                                newFeaturesCount={featuresCount}
                                regressionMetrics={regressionMetrics}
                                classifierResults={classifierResults}
                                classifierVisualization={classifierVisualization}
                            />

                            <div className="button-group">
                                <LoadDataButton onSummaryLoaded={handleSummaryLoaded} />
                                <ValidateDataButton />
                                <GenerateFeaturesButton onFeaturesGenerated={setFeaturesCount} />
                                <GenerateLabelsButton onLabelsGenerated={setLabelSummary} />
                                <TrainRegressionModelButton onRegressionComplete={setRegressionMetrics} />
                                <VisualizeRegressionButton /> {/* Visualize Regression */}
                                <GenerateClassifierLabelButton /> {/* New Generate Classifier Label Button */}
                                <TrainClassifiersButton onClassificationComplete={setClassifierResults} />
                                <VisualizeClassifiersButton onVisualizationComplete={setClassifierVisualization} />
                                <StartSimulationButton />
                                <RestartSimulationButton onRestart={handleSimulationRestart} />
                            </div>
                        </div>
                    </div>
                } />
                <Route path="/simulation" element={<Simulation3 />} />
            </Routes>
        </Router>
    );
}

export default App;
