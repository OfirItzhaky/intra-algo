import React, { useEffect, useState, useRef, useMemo } from "react";
import ReactDOM from "react-dom";
import { format } from "d3-format";
import { timeFormat } from "d3-time-format";
import GenerateNewBarButton from "./generate_new_bar_button"; // ✅ Import the button
import { scaleTime } from "d3-scale";

import {
  discontinuousTimeScaleProviderBuilder,
  Chart,
  ChartCanvas,
  CandlestickSeries,
  LineSeries,
  XAxis,
  YAxis,
  CrossHairCursor,
  CurrentCoordinate,
  SingleValueTooltip,
  ZoomButtons,
  OHLCTooltip,
  MouseCoordinateY, // ✅ Added missing import
  lastVisibleItemBasedZoomAnchor
} from "react-financial-charts";

const Simulation2 = () => {
  const [isFirstBarGenerated, setIsFirstBarGenerated] = useState(false);
  const [simulationData, setSimulationData] = useState([]); // ✅ Store API data
  const [visibleData, setVisibleData] = useState([]);

  // Move ScaleProvider setup into useMemo
  const { data: scaledData, xScale, xAccessor, displayXAccessor } = useMemo(() => {
    const ScaleProvider = discontinuousTimeScaleProviderBuilder()
      .inputDateAccessor(d => d.date instanceof Date ? d.date : new Date(d.date));
    return ScaleProvider(visibleData);
  }, [visibleData]);

  // Add debug log here
  console.log("DEBUG - Scale Provider:", {
    firstItem: visibleData[0],
    dateType: visibleData[0]?.date instanceof Date ? 'Date object' : typeof visibleData[0]?.date,
    xAccessorResult: visibleData[0] ? xAccessor(visibleData[0]) : 'no data'
  });

  useEffect(() => {
    fetch("http://localhost:8000/initialize-simulation/")
      .then(response => response.json())
      .then(data => {
        console.log("📊 Received Simulation Data:", data);

        if (data.status === "success" && data.data.length > 0) {
          const parsedData = data.data.map(d => ({
            date: new Date(`${d.Date} ${d.Time}:00`),  // Convert to Date object immediately
            open: d.Open,
            high: d.High,
            low: d.Low,
            close: d.Close,
            actualHigh: d.Actual_High,
            predictedHigh: d.Predicted_High,
            rf: d.RandomForest,
            lt: d.LightGBM,
            xg: d.XGBoost
          }));

          console.log("✅ Sample Processed Data:", parsedData.slice(0, 5));

          setSimulationData(parsedData);
          setVisibleData(parsedData);
          console.log("✅ After setting visibleData:", parsedData);

          console.log("📌 Updated `simulationData`:", parsedData);
          console.log("📌 Updated `visibleData`:", parsedData);

          console.log("Sample date string:", parsedData[0].date);
          console.log("Parsed date:", new Date(parsedData[0].date));

          console.log("DEBUG - Data Types:", {
            firstItemDate: parsedData[0].date,
            firstItemDateType: typeof parsedData[0].date,
            firstItemParsedDate: new Date(parsedData[0].date),
            isValidDate: !isNaN(new Date(parsedData[0].date).getTime())
          });
        } else {
          console.error("⚠️ No valid data received:", data);
        }
      })
      .catch(error => console.error("🚨 Error fetching simulation data:", error));
  }, []);

  const height = 700;
  const width = 1400;
  const margin = { left: 0, right: 48, top: 0, bottom: 24 };

  const predictedHighAccessor = (d) => d.predictedHigh;
  const actualHighAccessor = (d) => d.actualHigh;

  const chartRef = useRef(null);
  const pricesDisplayFormat = format(".2f");

    console.log("📌 Debugging visibleData before setting xExtents:", visibleData);
    console.log("📌 visibleData.length:", visibleData.length);


      // ✅ Debugging: Ensure visibleData has valid dates before using xAccessor
    console.log("📌 Checking Last Item in visibleData:", visibleData[visibleData.length - 1]);
    console.log("📌 Checking 100th Item in visibleData:", visibleData[Math.max(0, visibleData.length - 100)]);
    console.log("📌 Checking visibleData structure:", JSON.stringify(visibleData.slice(0, 5), null, 2));

    // ✅ Debugging: Ensure xAccessor values before setting xExtents
    const max = xAccessor(visibleData[visibleData.length - 1]);
    const min = xAccessor(visibleData[Math.max(0, visibleData.length - 100)]);

console.log("📌 Checking Last Item in visibleData:", visibleData[visibleData.length - 1]);
console.log("📌 Checking 100th Item in visibleData:", visibleData[Math.max(0, visibleData.length - 100)]);
console.log("📌 max:", max);
console.log("📌 min:", min);

// ✅ Ensure `xExtents` are valid
const xExtents = [min, max + 5];

console.log("📌 xExtents:", xExtents);

  // ✅ Debugging: Check if we have valid candlestick data
  visibleData.forEach((d, i) => {
    if (!d.open || !d.high || !d.low || !d.close) {
      console.warn(`❌ Malformed candlestick at index ${i}:`, d);
    }
  });
  // ✅ Debugging: Check xAccessor values before setting xExtents
visibleData.forEach((d, i) => {
    console.log(`xAccessor(${i}):`, xAccessor(d));
});

  const barChartExtents = (data) => {
    return data.volume;
  };

  const candleChartExtents = (data) => {
    return [data.high, data.low];
  };

  console.log("🔍 Final Processed Data for Chart:", visibleData);

  const volumeColor = (data) => {
    return data.close > data.open
      ? "rgba(38, 166, 154, 0.3)"
      : "rgba(239, 83, 80, 0.3)";
  };

  const volumeSeries = (data) => {
    return data.volume;
  };

  const openCloseColor = (data) => {
    return data.close > data.open ? "#26a69a" : "#ef5350";
  };
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
    }
  return (
    <div style={{ marginTop: "150px" }}>
      {/* ✅ Debug log before rendering ChartCanvas */}
      {console.log("📊 Final Visible Data for Chart:", visibleData)}

      <ChartCanvas
        height={height}
        ratio={3}
        width={width}
        margin={margin}
        data={visibleData}  // ✅ Confirm this is not empty
        displayXAccessor={displayXAccessor}
        seriesName="Data"
        xScale={xScale}
        xAccessor={xAccessor}
        xExtents={xExtents}
        zoomAnchor={lastVisibleItemBasedZoomAnchor}
      >
        <Chart id={3} height={height - margin.bottom} yExtents={d => [d.high, d.low]}>
          <XAxis
            showGridLines
            strokeStyle="white"
            tickLabelFill="white"
            gridLinesStrokeStyle="#444444"
          />
          <YAxis
            showGridLines
            strokeStyle="white"
            tickLabelFill="white"
            gridLinesStrokeStyle="#444444"
          />
          <CandlestickSeries />
          <LineSeries yAccessor={predictedHighAccessor} strokeStyle="red" />
          <CurrentCoordinate yAccessor={predictedHighAccessor} fillStyle="red" />
          <LineSeries yAccessor={actualHighAccessor} strokeStyle="blue" />
          <CurrentCoordinate yAccessor={actualHighAccessor} fillStyle="blue" />
          <MouseCoordinateY rectWidth={margin.right} displayFormat={pricesDisplayFormat} />
          <SingleValueTooltip origin={[8, 24]} textFill="white" yAccessor={predictedHighAccessor} label="Predicted High" valueFill="red" />
          <SingleValueTooltip origin={[8, 44]} textFill="white" yAccessor={actualHighAccessor} label="Actual High" valueFill="blue" />
          <ZoomButtons />
          <OHLCTooltip textFill="white" origin={[8, 16]} />
        </Chart>

        <CrossHairCursor />
      </ChartCanvas>

      <div className="button-container" style={{ marginTop: "5px" }}>
        <GenerateNewBarButton
          onNewBarGenerated={(newBar) => {
            console.log("➕ New Bar Generated (Raw):", newBar);
            const formattedNewBar = { ...newBar, date: new Date(newBar.date) };

            setSimulationData(prevData => {
              let updatedData = [...prevData, formattedNewBar];
              if (updatedData.length > 1) {
                updatedData[updatedData.length - 2].actualHigh = formattedNewBar.high;
              }
              console.log("📊 Updated Simulation Data (AFTER SET):", updatedData);
              return updatedData;
            });

            setVisibleData(prevData => {
              let updatedVisibleData = [...prevData, formattedNewBar];
              console.log("📊 Updated Visible Data (AFTER SET):", updatedVisibleData);
              return [...updatedVisibleData];
            });
            }
          }
          isFirstBarGenerated={isFirstBarGenerated}
          setIsFirstBarGenerated={setIsFirstBarGenerated}
        />
      </div>
    </div>
  );
};

export default Simulation2;