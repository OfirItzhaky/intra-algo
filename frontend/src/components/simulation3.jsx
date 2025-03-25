import React, { useEffect, useState, useRef } from "react";
import { plotDataLengthBarWidth } from "react-financial-charts";
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
  withSize,
  Label,
  Annotate,
  LabelAnnotation
} from "react-financial-charts";

import { initialData, PredActualData, classifierData } from "./initialData";
import axios from 'axios';
import GenerateNewBarButton from "./generate_new_bar_button"; // ✅ Import the button

const Simulation3 = () => {
  const ScaleProvider = discontinuousTimeScaleProviderBuilder().inputDateAccessor(
  d => new Date(d.date)  // ✅ This tells it to use real Date objects internally
);
initialData.forEach(d => {
  if (!(d.date instanceof Date)) {
    d.date = new Date(d.date);
  }
});
const { data, xScale, xAccessor, displayXAccessor } = ScaleProvider(initialData);





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
      const match = initialData.find((d) => {
        const dateA = d.date instanceof Date ? d.date : new Date(d.date);
        const dateB = p.date instanceof Date ? p.date : new Date(p.date);
        return dateA.getTime() === dateB.getTime();
      });

      if (match) {
        match.predictedHigh = p.predictedHigh;
        match.actualHigh = p.actualHigh;
      }
    });




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

const lastN = 20;
const from = new Date(displayXAccessor(data[data.length - lastN]).getTime() - 3 * 5 * 60 * 1000);
const to = new Date(displayXAccessor(data[data.length - 1]).getTime() + 3 * 5 * 60 * 1000);
const xExtents = [from, to];








console.log("📅 Final xExtents Dates", {
  from: xExtents[0].toISOString(),
  to: xExtents[1].toISOString(),
});

  // Log the final xExtents values
  console.log("📊 xExtents :", {
    xExtents

  });

  const gridHeight = height - margin.top - margin.bottom;

  const gridColor = "#555"; // Greyish color for the grid
  const axisColor = "#fff"; // White color for the axes

  const textColor = "#fff"; // White color for text

  const yExtents = (data) => {
    return [data.high, data.low];
  };
const dateTimeFormat = "%H:%M";  // Just hours and minutes
const timeDisplayFormat = timeFormat(dateTimeFormat);



const candleChartExtents = (d) => {
  const values = [d.high, d.low];
  if (d.predictedHigh !== undefined) values.push(d.predictedHigh);
  if (d.actualHigh !== undefined) values.push(d.actualHigh);
  return values;
};

const debugTickFormat = (d) => {
  console.log("🕒 Tick value:", d);
  return timeFormat("%H:%M")(d);
};

  const yEdgeIndicator = (data) => {
    return data.close;
  };

const predictedAnnotations = data
  .filter((d) => d.predictedHigh !== undefined)
  .map((d) => ({
    date: d.date,
    y: d.predictedHigh,
    label: d.predictedHigh.toFixed(2),
  }));

const actualAnnotations = data
  .filter((d) => d.actualHigh !== undefined)
  .map((d) => ({
    date: d.date,
    y: d.actualHigh,
    label: d.actualHigh.toFixed(2),
  }));


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

  // Add debug logs for data
  console.log("🔍 Chart Data Overview:", {
    totalDataPoints: data.length,
    firstPoint: data[0],
    lastPoint: data[data.length - 1],
    hasActualHigh: data.some(d => d.actualHigh !== undefined),
    hasPredictedHigh: data.some(d => d.predictedHigh !== undefined)
  });
console.log("📍 Predicted labels to annotate:", data.filter(d => d.predictedHigh != null).map(d => ({
  date: d.date,
  value: d.predictedHigh
})));
console.log("📍 Actual labels to annotate:", data.filter(d => d.actualHigh != null).map(d => ({
  date: d.date,
  value: d.actualHigh
})));
console.log("🔍 Sample xAccessor output (should be index or number):", xAccessor(data[0]));
console.log("📅 Is xAccessor a Date?", xAccessor(data[0]) instanceof Date);  // ❌ Likely false
console.log("🕓 Actual Date string from data:", data[0].date);
console.log("📆 displayXAccessor (should be Date):", displayXAccessor(data[0]));
console.log("📅 Is displayXAccessor returning a Date?", displayXAccessor(data[0]) instanceof Date); // ✅ Should be true
console.log("📏 Grid Height:", gridHeight);
console.log("🧠 xAccessor value:", xAccessor(data[0]));
console.log("🧠 displayXAccessor value:", displayXAccessor(data[0]));
console.log("📅 xExtents:", xExtents.map(x => x instanceof Date ? x.toISOString() : x));




  return (
    <div className="chart-container">
      <div className="chart-inner-container">
          <div style={{ color: 'white', fontSize: 16, padding: '10px' }}>
              🧪 Chart loaded with {data.length} bars
            </div>

        <ChartCanvas
          height={height}
          ratio={3}
          width={width}
          margin={margin}
          data={data}
          displayXAccessor={displayXAccessor}
          seriesName="Data"
          xScale={xScale}
          xAccessor={d => d.date}
          xExtents={xExtents}
          zoomAnchor={lastVisibleItemBasedZoomAnchor}
        >

          <Chart id={3} height={gridHeight} yExtents={candleChartExtents}>
            <XAxis
            tickValues={data.map(d => d.date)}
              tickLabelFill="#ffffff"         // ✅ White text
              strokeStyle="#ffffff"           // ✅ White axis line
                tickStrokeStyle="#ffffff"
                  showTickLabel={true}
 ticks={8} // ✅ Reduce number of ticks

              tickFormat={(d) => timeFormat("%H:%M")(new Date(d))}
              tickLabelAngle={15}            // ✅ Rotate for visibility
            />

/>

            <YAxis
              showGridLines
              gridLinesStrokeStyle={gridColor}
              strokeStyle={axisColor}
              tickLabelFill={axisColor}
              tickFormat={pricesDisplayFormat}
            />


            <CandlestickSeries
              fill={(d) => (d.close > d.open ? "#26a69a" : "#ef5350")}
              stroke="#ffffff"  // white border for contrast
              wickStroke="white"                     // white wick
              candleStrokeWidth={0.6}   // thinner border
              wickStrokeWidth={0.6}
              width={() => 10}                       // 🔥 force candle width
            />
            <LineSeries yAccessor={predictedLine.accessor} strokeStyle={predictedLine.stroke} />
            <LineSeries yAccessor={actualLine.accessor} strokeStyle={actualLine.stroke} />
            <CurrentCoordinate
              yAccessor={predictedLine.accessor}
              fillStyle={predictedLine.stroke}
            />
            <CurrentCoordinate
              yAccessor={actualLine.accessor}
              fillStyle={actualLine.stroke}
            />
            <MouseCoordinateY
              rectWidth={margin.right}
              displayFormat={pricesDisplayFormat}
            />
            <MouseCoordinateX
              displayFormat={timeDisplayFormat}
            />

            <EdgeIndicator
              itemType="last"
              orient="right"
              edgeAt="right"
              yAccessor={predictedLine.accessor}
              fill="red"
              lineStroke="red"
              displayFormat={format(".2f")}
            />

            <EdgeIndicator
              itemType="last"
              orient="right"
              edgeAt="right"
              yAccessor={actualLine.accessor}
              fill="blue"
              lineStroke="blue"
              displayFormat={format(".2f")}
            />
             <Annotate
              with={LabelAnnotation}
              when={d => d.predictedHigh !== undefined}
              usingProps={{
                yAccessor: d => d.predictedHigh,
                    y: d => d.predictedHigh + 1.5,   // ⬆️ 1.5 points above the line
                fill: "red",
                text: d => d.predictedHigh.toFixed(2),
                fontSize: 11,
                textAnchor: "middle"
              }}
            />


              <Annotate
                  with={LabelAnnotation}
                  when={d => d.actualHigh !== undefined}
                  usingProps={{
                    yAccessor: d => d.actualHigh,
                    fill: "blue",
                    text: d => d.actualHigh.toFixed(2),
                    fontSize: 11,
                    textAnchor: "middle",
                    dy: -10  // ⬆️ move label upward by 10px from the line
                  }}
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
