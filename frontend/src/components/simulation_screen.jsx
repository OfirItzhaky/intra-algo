import React, { useEffect, useState } from "react";
import "../index.css"; // ✅ Uses the correct styles
import { CircleMarker } from "react-financial-charts";
import { ScatterSeries, Annotate } from "react-financial-charts";

import {
    ChartCanvas,
    Chart,
    CandlestickSeries,
    LineSeries,
    XAxis,
    YAxis
} from "react-financial-charts";
import { scaleTime } from "d3-scale";
import GenerateNewBarButton from "./generate_new_bar_button"; // ✅ Import the button

function SimulationScreen() {
    const [simulationData, setSimulationData] = useState([]);
    const [visibleData, setVisibleData] = useState([]);
    const [isFirstBarGenerated, setIsFirstBarGenerated] = useState(false);

    useEffect(() => {
        fetch("http://localhost:8000/initialize-simulation/")
            .then(response => response.json())
            .then(data => {
                console.log("📊 Received Simulation Data:", data);

                if (data.status === "success" && data.data.length > 0) {
                    const parsedData = data.data.map(d => ({
                        date: new Date(d.Date + " " + d.Time),
                        open: d.Open,
                        high: d.High,
                        low: d.Low,
                        close: d.Close,
                        volume: d.Volume,
                        actualHigh: d.Actual_High,
                        predictedHigh: d.Predicted_High,
                    }));

                    console.log("✅ Sample Processed Data:", parsedData.slice(0, 5));

                    setSimulationData(parsedData);
                    setVisibleData(parsedData);
                } else {
                    console.error("⚠️ No valid data received:", data);
                }
            })
            .catch(error => console.error("🚨 Error fetching simulation data:", error));
    }, []);

    const handleNewBarGenerated = (newBar) => {
        console.log("➕ New Bar Generated:", newBar);

        setSimulationData(prevData => {
            let updatedData = [...prevData, newBar];

            // ✅ Ensure actual high is only added for previous bar
            if (updatedData.length > 1) {
                updatedData[updatedData.length - 2].actualHigh = newBar.high; // Attach actual high to previous bar
            }

            return updatedData;
        });

        setVisibleData(prevData => [...prevData, newBar]);
    };

    console.log("📊 Sample of visibleData for Chart:", visibleData.slice(0, 5));

    if (visibleData.length === 0) {
        console.warn("⚠️ No visible data available for the chart.");
        return <p style={{ color: 'white' }}>⚠️ No Data Available! Check Console Logs.</p>;
    }

    return (
        <div className="simulation-container">
            <h2 style={{ color: "white" }}>📈 Simulation Running...</h2>
            <p style={{ color: "lightgray" }}>Click "Generate Next Bar" to proceed.</p>

            <div className="chart-scroll-container">
                <div className="chart-inner-container">
                    <ChartCanvas
                        height={400}
                        width={1600}
                        ratio={3}
                        data={visibleData}
                        seriesName="CandlestickChart"
                        xAccessor={(d) => d.date}
                        xScale={scaleTime()}
                        xExtents={[
                            visibleData[Math.max(0, visibleData.length - 20)].date,
                            visibleData[visibleData.length - 1].date
                        ]}
                    >
                        <Chart id={1} yExtents={(d) => [d.high, d.low]}>
                            <XAxis strokeStyle="white" tickLabelFill="white" />
                            <YAxis strokeStyle="white" tickLabelFill="white" />
                            <CandlestickSeries />

                            {/* ✅ Overlay Actual & Predicted High */}
                            <LineSeries
                                yAccessor={(d, i) => (i === visibleData.length - 1 ? null : d.actualHigh)}
                                strokeStyle="blue"
                                strokeWidth={2}
                                marker={CircleMarker}
                                markerProps={{ stroke: "blue", fill: "blue", r: 3 }}
                            />

                            <LineSeries
                                yAccessor={(d) => d.predictedHigh}
                                strokeStyle="red"
                                strokeWidth={2}
                                strokeDasharray="5,5"
                                marker={CircleMarker}
                                markerProps={{ stroke: "red", fill: "red", r: 3 }}
                            />

                            {/* ✅ ScatterSeries for predictions */}
                            <ScatterSeries
                                yAccessor={(d, i) => (i === visibleData.length - 1 ? null : d.actualHigh)}
                                marker={CircleMarker}
                                markerProps={{ stroke: "blue", fill: "blue", r: 3 }}
                            />

                            <ScatterSeries
                                yAccessor={(d) => d.predictedHigh}
                                marker={CircleMarker}
                                markerProps={{ stroke: "red", fill: "red", r: 4 }}
                            />

                            {/* ✅ Remove Actual High Labels for Last Bar */}
                            {visibleData.map((d, i) =>
                                i === visibleData.length - 1 ? null : (
                                    <Annotate
                                        key={`actual-${i}`}
                                        with={(props) => (
                                            <text
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(d.actualHigh)}
                                                textAnchor="middle"
                                                fontSize={12}
                                                fill="blue"
                                                dy={-10}
                                            >
                                                {d.actualHigh ? d.actualHigh.toFixed(2) : ""}
                                            </text>
                                        )}
                                        when={() => true}
                                    />
                                )
                            )}

                            {/* ✅ Keep Predicted High Labels */}
                            {visibleData.map((d, i) => (
                                <Annotate
                                    key={`predicted-${i}`}
                                    with={(props) => (
                                        <text
                                            x={props.xScale(props.xAccessor(d))}
                                            y={props.yScale(d.predictedHigh)}
                                            textAnchor="middle"
                                            fontSize={12}
                                            fill="red"
                                            dy={-10}
                                        >
                                            {d.predictedHigh.toFixed(2)}
                                        </text>
                                    )}
                                    when={() => true}
                                />
                            ))}
                        </Chart>
                    </ChartCanvas>
                </div>
            </div>

            {/* ✅ Button properly styled now */}
            <div className="button-container">
                <GenerateNewBarButton
                    onNewBarGenerated={handleNewBarGenerated}
                    isFirstBarGenerated={isFirstBarGenerated}
                    setIsFirstBarGenerated={setIsFirstBarGenerated}
                />

            </div>
        </div>
    );
}

export default SimulationScreen;
