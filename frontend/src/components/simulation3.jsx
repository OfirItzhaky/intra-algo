import React, { useEffect, useState, useRef } from "react";
import ReactDOM from "react-dom";
import { format } from "d3-format";
import { timeFormat } from "d3-time-format";
import {
  ema,
  discontinuousTimeScaleProviderBuilder,
  Chart,
  ChartCanvas,
  CurrentCoordinate,
  CandlestickSeries,

  LineSeries,
  MovingAverageTooltip,
  OHLCTooltip,
  SingleValueTooltip,
  lastVisibleItemBasedZoomAnchor,
  XAxis,
  YAxis,
  CrossHairCursor,
  EdgeIndicator,
  MouseCoordinateX,
  MouseCoordinateY,
  ZoomButtons,
  withDeviceRatio,
  withSize
} from "react-financial-charts";
import { initialData, PredActualData, classifierData } from "./initialData";
import axios from 'axios';
import GenerateNewBarButton from "./generate_new_bar_button"; // ✅ Import the button

const Simulation3 = () => {
  const ScaleProvider = discontinuousTimeScaleProviderBuilder().inputDateAccessor(
    (d) => new Date(d.date)
  );
  const [isFirstBarGenerated, setIsFirstBarGenerated] = useState(false);

  const height = 700;
  const width = 900;
  const margin = { left: 0, right: 48, top: 0, bottom: 24 };
    const predictedLine = {
  accessor: (d) => (d ? d.predictedHigh : null),
  stroke: "red",
    options: () => ({ windowSize: 1 })  // 👈 just a placeholder

};
const actualLine = {
  accessor: (d) => (d ? d.actualHigh : null),
  stroke: "blue",
    options: () => ({ windowSize: 1 })  // 👈 just a placeholder

};

    // ✅ Merge predicted and actual highs into initialData before passing to ScaleProvider
    PredActualData.forEach((p) => {
      const match = initialData.find((d) => d.date === p.date);
      if (match) {
        match.predictedHigh = p.predictedHigh;
        match.actualHigh = p.actualHigh;
      }
    });

  const { data, xScale, xAccessor, displayXAccessor } = ScaleProvider(
    initialData
  );

  console.log("Processed Data:", data);
  console.log("xScale:", xScale);
  console.log("xAccessor for first data point:", xAccessor(data[0]));
  console.log("Date from xAccessor for first data point:", new Date(data[0].date));
  console.log("Display X Accessor for first data point:", displayXAccessor(data[0]));
  console.log("xAccessor for last data point:", xAccessor(data[data.length - 1]));
  console.log("Date from xAccessor for last data point:", new Date(data[data.length - 1].date));
  console.log("First data point:", data[0]);
  console.log("Last data point:", data[data.length - 1]);
  console.log("xAccessor for first data point:", xAccessor(data[0]));
  console.log("xAccessor for last data point:", xAccessor(data[data.length - 1]));

  const pricesDisplayFormat = format(".2f");
  const max = xAccessor(data[data.length - 1]);
  const min = xAccessor(data[Math.max(0, data.length - 100)]);
  
  // Add debug logging
  console.log("🎯 xAccessor values:", {
    min: min,
    max: max,
    dataLength: data.length,
    firstDate: new Date(data[0].date),
    lastDate: new Date(data[data.length - 1].date)
  });

  const rightPadding = (max - min) * 0.1;
  const xExtents = [min, max + rightPadding];

  // Log the final xExtents values
  console.log("📊 xExtents with padding:", {
    start: new Date(min),
    end: new Date(max + rightPadding),
    paddingAmount: rightPadding
  });

  const gridHeight = height - margin.top - margin.bottom;

  const gridColor = "#555"; // Greyish color for the grid
  const axisColor = "#fff"; // White color for the axes

  const textColor = "#fff"; // White color for text

  const yExtents = (data) => {
    return [data.high, data.low];
  };
  const dateTimeFormat = "%d %b %H:%M:%S"; // Day, Month, Hour:Minute:Second
  const timeDisplayFormat = timeFormat(dateTimeFormat);



  const candleChartExtents = (data) => {
    return [data.high, data.low];
  };

  const yEdgeIndicator = (data) => {
    return data.close;
  };



  const openCloseColor = (data) => {
    return data.close > data.open ? "#26a69a" : "#ef5350";
  };

  console.log("Initial Data:", initialData);

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

  return (
    <div className="chart-container">
      <div className="chart-inner-container">
        <ChartCanvas
          height={height}
          ratio={3}
          width={width}
          margin={margin}
          data={data}
          displayXAccessor={displayXAccessor}
          seriesName="Data"
          xScale={xScale}
          xAccessor={xAccessor}
          xExtents={xExtents}
          zoomAnchor={lastVisibleItemBasedZoomAnchor}
        >

          <Chart id={3} height={gridHeight} yExtents={candleChartExtents}>
            <XAxis
              showGridLines
              gridLinesStrokeStyle={gridColor}
              strokeStyle={axisColor}
              tickLabelFill={axisColor}
              tickFormat={timeDisplayFormat}
              showTickLabel={true}
              ticks={10}
              tickLabelAngle={-45}
            />
            <YAxis
              showGridLines
              gridLinesStrokeStyle={gridColor}
              strokeStyle={axisColor}
              tickLabelFill={axisColor}
              tickFormat={pricesDisplayFormat}
            />
            <CandlestickSeries />
            <LineSeries yAccessor={predictedLine.accessor} strokeStyle={predictedLine.stroke} />
            <CurrentCoordinate
              yAccessor={predictedLine.accessor}
              fillStyle={predictedLine.stroke}
            />
            <LineSeries yAccessor={actualLine.accessor} strokeStyle={actualLine.stroke} />
            <CurrentCoordinate
              yAccessor={actualLine.accessor}
              fillStyle={actualLine.stroke}
            />
            <MouseCoordinateY
              rectWidth={margin.right}
              displayFormat={pricesDisplayFormat}
            />
            <EdgeIndicator
              itemType="last"
              rectWidth={margin.right}
              fill={openCloseColor}
              lineStroke={openCloseColor}
              displayFormat={pricesDisplayFormat}
              yAccessor={yEdgeIndicator}
            />
            <MovingAverageTooltip
              origin={[8, 24]}
              textFill={textColor}
              options={[
                {
                  yAccessor: predictedLine.accessor,
                  type: "Predicted",
                  stroke: predictedLine.stroke,
                  windowSize: predictedLine.options().windowSize
                },
                {
                  yAccessor: actualLine.accessor,
                  type: "Actual",
                  stroke: actualLine.stroke,
                  windowSize: actualLine.options().windowSize
                }
              ]}
            />

            <ZoomButtons />
            <OHLCTooltip
              origin={[8, 16]}
              textFill={textColor}
            />
          </Chart>

          <CrossHairCursor />
        </ChartCanvas>

        <div className="button-container">
                <GenerateNewBarButton
                    onNewBarGenerated={handleNewBarGenerated}
                    isFirstBarGenerated={isFirstBarGenerated}
                    setIsFirstBarGenerated={setIsFirstBarGenerated}
                />
            </div>
      </div>
    </div>
  );
};

export default Simulation3;
