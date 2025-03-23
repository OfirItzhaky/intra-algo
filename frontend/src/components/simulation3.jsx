import React from "react";
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

const Simulation3 = () => {
  const ScaleProvider = discontinuousTimeScaleProviderBuilder().inputDateAccessor(
    (d) => new Date(d.date)
  );
  const height = 700;
  const width = 900;
  const margin = { left: 0, right: 48, top: 0, bottom: 24 };
    const predictedLine = {
  accessor: (d) => d.predictedHigh,
  stroke: "red",
    options: () => ({ windowSize: 1 })  // 👈 just a placeholder

};
const actualLine = {
  accessor: (d) => d.actualHigh,
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

  const pricesDisplayFormat = format(".2f");
  const max = xAccessor(data[data.length - 1]);
  const min = xAccessor(data[Math.max(0, data.length - 100)]);
  const xExtents = [min, max + 5];

  const gridHeight = height - margin.top - margin.bottom;

  const gridColor = "#555"; // Greyish color for the grid
  const axisColor = "#fff"; // White color for the axes

  const textColor = "#fff"; // White color for text

  const yExtents = (data) => {
    return [data.high, data.low];
  };
  const dateTimeFormat = "%d %b";
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

  return (
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
          showTickLabel={false}
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
  );
};

export default Simulation3;
