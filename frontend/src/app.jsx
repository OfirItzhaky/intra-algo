import React from 'react';
import './index.css';
import LoadDataButton from './components/load_data_button';
import ValidateDataButton from './components/validate_data_button';
import GenerateFeaturesButton from './components/generate_features_button';
import StartSimulationButton from './components/start_simulation_button';
import TrainRegressionModelButton from './components/train_regression_model_button';
import TrainClassifiersButton from './components/train_classifiers_button';
import RestartSimulationButton from './components/restart_simulation_button';

function App() {
    return (
        <div className="app-container">
            <div className="content-box">
                <h1 className="app-title">Intra Algo Trading Platform</h1>
                <p className="app-subtitle">Intra Algo Simulator - Version 1.0</p>

                {/* Buttons Area */}
                <div className="button-group">
                    <LoadDataButton />
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
