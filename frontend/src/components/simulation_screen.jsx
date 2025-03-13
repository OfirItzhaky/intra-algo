import React, { useEffect, useState } from "react";
import "../index.css";  // ✅ Adjusted for correct path
import { CircleMarker } from "react-financial-charts";
import { ScatterSeries, Annotate, LabelAnnotation } from "react-financial-charts"; // ✅ Correct imports

import {
    ChartCanvas,
    Chart,
    CandlestickSeries,
    LineSeries,  // ✅ For Actual & Predicted High
    XAxis,
    YAxis
} from "react-financial-charts";
import { scaleTime } from "d3-scale";

function SimulationScreen() {
    const [simulationData, setSimulationData] = useState([]);
    const [visibleData, setVisibleData] = useState([]);

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
                        actualHigh: d.Actual_High,  // ✅ Add Actual High
                        predictedHigh: d.Predicted_High, // ✅ Add Predicted High
                    }));

                    console.log("✅ Sample Processed Data:", parsedData.slice(0, 5));

                    setSimulationData(parsedData);
                    setVisibleData(parsedData);  // ✅ Show all data, but default view will be last 20 bars
                } else {
                    console.error("⚠️ No valid data received:", data);
                }
            })
            .catch(error => console.error("🚨 Error fetching simulation data:", error));
    }, []);

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
                                yAccessor={(d) => d.actualHigh}
                                strokeStyle="blue" // ✅ Corrected strokeStyle
                                strokeWidth={2}
                                marker={CircleMarker}
                                markerProps={{ strokeStyle: "blue", fill: "blue", r: 3 }}
                            />

                            <LineSeries
                                yAccessor={(d) => d.predictedHigh}
                                strokeStyle="red" // ✅ Corrected strokeStyle
                                strokeWidth={2}
                                strokeDasharray="5,5"
                                marker={CircleMarker}
                                markerProps={{ strokeStyle: "red", fill: "red", r: 3 }}
                            />

                            {/* ✅ Dots for Actual High */}
                            <ScatterSeries
                                yAccessor={(d) => d.actualHigh}
                                marker={CircleMarker}
                                markerProps={{ strokeStyle: "blue", fill: "blue", r: 4 }}
                            />

                            {/* ✅ Dots for Predicted High */}
                            <ScatterSeries
                                yAccessor={(d) => d.predictedHigh}
                                marker={CircleMarker}
                                markerProps={{ strokeStyle: "red", fill: "red", r: 4 }}
                            />

                            {/* ✅ Labels for Actual High */}
                            {visibleData.map((d, i) => (
                                <Annotate
                                    key={`actual-${i}`}
                                    with={LabelAnnotation}
                                    when={() => true}
                                    usingProps={{
                                        x: d.date,
                                        y: d.actualHigh,
                                        text: d.actualHigh.toFixed(2),
                                        textAnchor: "middle",
                                        fontSize: 12,
                                        fill: "blue",
                                        opacity: 1,
                                        dx: -10,
                                        dy: -10
                                    }}
                                />
                            ))}

                            {/* ✅ Labels for Predicted High */}
                            {visibleData.map((d, i) => (
                                <Annotate
                                    key={`predicted-${i}`}
                                    with={LabelAnnotation}
                                    when={() => true}
                                    usingProps={{
                                        x: d.date,
                                        y: d.predictedHigh,
                                        text: d.predictedHigh.toFixed(2),
                                        textAnchor: "middle",
                                        fontSize: 12,
                                        fill: "red",
                                        opacity: 1,
                                        dx: 10,
                                        dy: -10
                                    }}
                                />
                            ))}

                        </Chart>
                    </ChartCanvas>
                </div>
            </div>

            <button className="generate-bar-button">
                ➕ Generate Next Bar
            </button>
        </div>
    );
}

export default SimulationScreen;
