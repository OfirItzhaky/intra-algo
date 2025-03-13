import React, { useEffect, useState } from "react";
import "../index.css";  // ✅ Adjusted to go one folder up

import {
    ChartCanvas,
    Chart,
    CandlestickSeries,
    XAxis,
    YAxis
} from "react-financial-charts";
import { scaleTime } from "d3-scale";

function SimulationScreen() {
    const [simulationData, setSimulationData] = useState([]);
    const [visibleData, setVisibleData] = useState([]); // ✅ For controlling view

    useEffect(() => {
        fetch("http://localhost:8000/initialize-simulation/")
            .then(response => response.json())
            .then(data => {
                console.log("📊 Received Simulation Data:", data);  // ✅ Debugging API Response

                if (data.status === "success" && data.data.length > 0) {
                    const parsedData = data.data.map(d => ({
                        date: new Date(d.Date + " " + d.Time), // ✅ Ensure Date object
                        open: d.Open,
                        high: d.High,
                        low: d.Low,
                        close: d.Close,
                        volume: d.Volume
                    }));

                    console.log("✅ Sample Processed Data:", parsedData.slice(0, 5)); // ✅ Debugging

                    setSimulationData(parsedData);
                    setVisibleData(parsedData.slice(-20)); // ✅ Show last 20 initially
                } else {
                    console.error("⚠️ No valid data received:", data);
                }
            })
            .catch(error => console.error("🚨 Error fetching simulation data:", error));
    }, []);

    // ✅ Print a sample of `visibleData` to debug
    console.log("📊 Sample of visibleData for Chart:", visibleData.slice(0, 5));

    // ✅ Check if data is empty and show error message
    if (visibleData.length === 0) {
        console.warn("⚠️ No visible data available for the chart.");
        return <p style={{ color: 'white' }}>⚠️ No Data Available! Check Console Logs.</p>;
    }

   return (
    <div className="simulation-container">
        <h2 style={{ color: "white" }}>📈 Simulation Running...</h2>
        <p style={{ color: "lightgray" }}>Click "Generate Next Bar" to proceed.</p>

        {/* ✅ Scrollable Wrapper for the Chart */}
        {/* ✅ Scrollable Wrapper for the Chart */}
        {/* ✅ Scrollable Wrapper for the Chart */}
        <div className="chart-scroll-container">
            <div className="chart-inner-container">
                <ChartCanvas
                    height={400}
                    width={1600}  // ✅ Ensure a wide chart for scrolling
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
                    </Chart>
                </ChartCanvas>
            </div>
        </div>



        <button style={{ backgroundColor: "#ff9800", color: "black", fontWeight: "bold" }}>
            ➕ Generate Next Bar
        </button>
    </div>
    );
}

export default SimulationScreen;
