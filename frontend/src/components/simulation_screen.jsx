import React, { useEffect, useState, useRef } from "react";
import "../index.css"; // ✅ Uses the correct styles
import { CircleMarker } from "react-financial-charts";
import { Annotate } from "react-financial-charts";

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

// ✅ Custom SquareMarker Component
const SquareMarker = ({ x, y, value, label }) => {
    if (isNaN(y) || value === undefined) return null;  // ✅ Prevent NaN issues

    return (
        <g>
            <rect
                x={x - 8}
                y={y - 8}
                width={18}
                height={18}
                fill={value === 1 ? "#00FF00" : "#FF0000"} // ✅ Green for 1, Red for 0
                rx={3} // ✅ Rounded corners
            />
            <text
                x={x}
                y={y + 5}
                textAnchor="middle"
                fontSize="12px"
                fontWeight="bold"
                fill="white"
            >
                {label}
            </text>
        </g>
    );
};

function SimulationScreen() {
    const [simulationData, setSimulationData] = useState([]);
    const [visibleData, setVisibleData] = useState([]);
    const [chartKey, setChartKey] = useState(0);
    const [isFirstBarGenerated, setIsFirstBarGenerated] = useState(false);
    const chartRef = useRef(null);  // ✅ Reference to chart for scrolling

    // ✅ Auto-scroll to the rightmost part of the chart
    const scrollToRightMost = () => {
        if (chartRef.current) {
            chartRef.current.scrollLeft = chartRef.current.scrollWidth;
        }
    };

    useEffect(() => {
        console.log("🔄 Re-rendering Chart due to visibleData update...");
        setChartKey(prevKey => prevKey + 1);

        // ✅ Auto-scroll after updating visible data
        scrollToRightMost();
    }, [visibleData]);

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
                        rf: d.RandomForest,
                        lt: d.LightGBM,
                        xg: d.XGBoost
                    }));

                    console.log("✅ Sample Processed Data:", parsedData.slice(0, 5));

                    setSimulationData(parsedData);
                    setVisibleData(parsedData);

                    // ✅ Scroll to latest bars after loading
                    setTimeout(scrollToRightMost, 100);
                } else {
                    console.error("⚠️ No valid data received:", data);
                }
            })
            .catch(error => console.error("🚨 Error fetching simulation data:", error));
    }, []);

    const handleNewBarGenerated = (newBar) => {
        console.log("➕ New Bar Generated (Raw):", newBar);

        const formattedNewBar = {
            ...newBar,
            date: new Date(newBar.date)  // 🔹 Convert date string to JavaScript Date
        };

        setSimulationData(prevData => {
            let updatedData = [...prevData, formattedNewBar];

            // ✅ Ensure actual high is only added for the previous bar
            if (updatedData.length > 1) {
                updatedData[updatedData.length - 2].actualHigh = formattedNewBar.high;
            }

            console.log("📊 Updated Simulation Data (AFTER SET):", updatedData);
            return updatedData;
        });

        setVisibleData(prevData => {
            let updatedVisibleData = [...prevData, formattedNewBar];

            console.log("📊 Updated Visible Data (AFTER SET):", updatedVisibleData);
            console.log("📊 Last Bar in Visible Data:", updatedVisibleData[updatedVisibleData.length - 1]);

            setTimeout(scrollToRightMost, 100);  // ✅ Scroll after adding a new bar
            return [...updatedVisibleData];
        });
    };

    console.log("📊 Chart Rendering - Visible Data Length:", visibleData.length);
    console.log("📊 Chart Rendering - Last 5 Data Points:", visibleData.slice(-5));

    if (visibleData.length === 0) {
        console.warn("⚠️ No visible data available for the chart.");
        return <p style={{ color: 'white' }}>⚠️ No Data Available! Check Console Logs.</p>;
    }

    return (
        <div className="simulation-container">
            <h2 style={{ color: "white" }}>📈 Simulation Running...</h2>
            <p style={{ color: "lightgray" }}>Click "Generate Next Bar" to proceed.</p>

            <div className="chart-scroll-container" ref={chartRef}>
                <div className="chart-inner-container">
                    <ChartCanvas
                        key={chartKey}
                        height={400}
                        width={1600}
                        ratio={3}
                        data={visibleData}
                        seriesName="CandlestickChart"
                        xAccessor={(d) => d.date}
                        xScale={scaleTime()}
                        xExtents={[
                            visibleData.length > 3
                                ? visibleData[visibleData.length - 20].date  // ✅ Adds padding for extra bars
                                : visibleData[0].date,
                            new Date(visibleData[visibleData.length - 1].date.getTime() + 15 * 60 * 1000)  // ✅ Adds extra 15 min
                        ]}

                    >
                        <Chart id={1} yExtents={(d) => [Math.min(d.low, d.predictedHigh || d.low), Math.max(d.high, d.predictedHigh || d.high)]}>
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

                            {/* ✅ Classifier Predictions Below Candles */}
                            {visibleData.map((d, i) => (
                                <React.Fragment key={i}>
                                    <Annotate
                                        with={(props) => (
                                            <SquareMarker
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(d.low) + 30}
                                                value={d.rf}
                                                label="RF"
                                            />
                                        )}
                                        when={() => true}
                                    />
                                    <Annotate
                                        with={(props) => (
                                            <SquareMarker
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(d.low) + 60}
                                                value={d.lt}
                                                label="LT"
                                            />
                                        )}
                                        when={() => true}
                                    />
                                    <Annotate
                                        with={(props) => (
                                            <SquareMarker
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(d.low) + 90}
                                                value={d.xg}
                                                label="XG"
                                            />
                                        )}
                                        when={() => true}
                                    />
                                </React.Fragment>
                            ))}

                            {/* ✅ Keep Predicted High Labels */}
                            {visibleData.map((d, i) => (
                                <Annotate
                                    key={`predicted-${i}`}
                                    with={(props) => {
                                        const yValue = d.predictedHigh;
                                        return yValue !== undefined && !isNaN(yValue) ? (
                                            <text
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(yValue)}
                                                textAnchor="middle"
                                                fontSize={12}
                                                fill="red"
                                                dy={-10}
                                            >
                                                {yValue.toFixed(2)}
                                            </text>
                                        ) : null;
                                    }}
                                    when={() => true}
                                />
                            ))}
                            {visibleData.map((d, i) => (
                                <Annotate
                                    key={`actual-${i}`}
                                    with={(props) => {
                                        const yValue = d.actualHigh;
                                        return (
                                            yValue !== undefined &&
                                            !isNaN(yValue) &&
                                            i !== visibleData.length - 1 // ✅ Skip the last bar
                                        ) ? (
                                            <text
                                                x={props.xScale(props.xAccessor(d))}
                                                y={props.yScale(yValue) - 12}  // ✅ Shift slightly up to avoid overlap
                                                textAnchor="middle"
                                                fontSize={12}
                                                fill="blue"  // ✅ Blue for actual values
                                                dy={-10}
                                            >
                                                {yValue.toFixed(2)}
                                            </text>
                                        ) : null;
                                    }}
                                    when={() => true}
                                />
                            ))}

                        </Chart>
                    </ChartCanvas>
                </div>
            </div>

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
