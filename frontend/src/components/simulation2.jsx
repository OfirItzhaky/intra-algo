import React from "react";
import ReactDOM from "react-dom";
import { format } from "d3-format";
import { timeFormat } from "d3-time-format";
import {useState} from "react";
import GenerateNewBarButton from "./generate_new_bar_button"; // ✅ Import the button

import {
  elderRay,
  ema,
  discontinuousTimeScaleProviderBuilder,
  Chart,
  ChartCanvas,
  CurrentCoordinate,
  BarSeries,
  CandlestickSeries,
  ElderRaySeries,
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
import { initialData } from "./data";

const Simulation2 = () => {
   const [isFirstBarGenerated, setIsFirstBarGenerated] = useState(false);

  const ScaleProvider = discontinuousTimeScaleProviderBuilder().inputDateAccessor(
    (d) => new Date(d.date)
  );
  const height = 700;
  const width = 1400;
  const margin = { left: 0, right: 48, top: 0, bottom: 24 };

  const ema12 = ema()
    .id(1)
    .options({ windowSize: 12 })
    .merge((d, c) => {
      d.ema12 = c;
    })
    .accessor((d) => d.ema12);

  const ema26 = ema()
    .id(2)
    .options({ windowSize: 26 })
    .merge((d, c) => {
      d.ema26 = c;
    })
    .accessor((d) => d.ema26);

  const elder = elderRay();

  const calculatedData = elder(ema26(ema12(initialData)));
  const { data, xScale, xAccessor, displayXAccessor } = ScaleProvider(
    initialData
  );
  const pricesDisplayFormat = format(".2f");
  const max = xAccessor(data[data.length - 1]);
  const min = xAccessor(data[Math.max(0, data.length - 100)]);
  const xExtents = [min, max + 5];

  const gridHeight = height - margin.top - margin.bottom;

  const elderRayHeight = 100;
  const elderRayOrigin = (_, h) => [0, h - elderRayHeight];
  const barChartHeight = gridHeight / 4;
  const barChartOrigin = (_, h) => [0, h - barChartHeight - elderRayHeight];
  const chartHeight = gridHeight - elderRayHeight;
  const yExtents = (data) => {
    return [data.high, data.low];
  };
  const dateTimeFormat = "%d %b";
  const timeDisplayFormat = timeFormat(dateTimeFormat);

  const barChartExtents = (data) => {
    return data.volume;
  };

  const candleChartExtents = (data) => {
    return [data.high, data.low];
  };

  const yEdgeIndicator = (data) => {
    return data.close;
  };

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

        setVisibleData(prevData => {
            let updatedVisibleData = [...prevData, formattedNewBar];

            console.log("📊 Updated Visible Data (AFTER SET):", updatedVisibleData);
            console.log("📊 Last Bar in Visible Data:", updatedVisibleData[updatedVisibleData.length - 1]);

            setTimeout(scrollToRightMost, 100);  // ✅ Scroll after adding a new bar
            return [...updatedVisibleData];
        });
    };
  return (
    <div style={{ marginTop: "150px" }}>  {/* ✅ Added margin here */}

        <ChartCanvas
            height={height}
            ratio={3}
            width={width}
            margin={margin}
            data={data}   // USE VISIBLE DATA FROM MY CODE...MAKE THE FORMAT LIKE data.JS FILE
            displayXAccessor={displayXAccessor}
            seriesName="Data"
            xScale={xScale}
            xAccessor={xAccessor}
            xExtents={xExtents}
            zoomAnchor={lastVisibleItemBasedZoomAnchor}

        >

            <Chart id={3} height={chartHeight} yExtents={candleChartExtents}>
                <XAxis
                    showGridLines
                    strokeStyle="white"            // Axis line color
                    tickLabelFill="white"          // Tick label text color
                    gridLinesStrokeStyle="#444444" // Optional: softer gray gridlines
                />
                <YAxis
                    showGridLines
                    strokeStyle="white"
                    tickLabelFill="white"
                    gridLinesStrokeStyle="#444444"
                />
                <CandlestickSeries />
                <LineSeries yAccessor={ema26.accessor()} strokeStyle={ema26.stroke()} />
                <CurrentCoordinate
                    yAccessor={ema26.accessor()}
                    fillStyle={ema26.stroke()}
                />
                <LineSeries yAccessor={ema12.accessor()} strokeStyle={ema12.stroke()} />
                <CurrentCoordinate
                    yAccessor={ema12.accessor()}
                    fillStyle={ema12.stroke()}
                />
                <MouseCoordinateY
                    rectWidth={margin.right}
                    displayFormat={pricesDisplayFormat}
                />

                <MovingAverageTooltip
                    origin={[8, 24]}
                    textFill="white"
                    options={[
                        {
                            yAccessor: ema26.accessor(),
                            type: "EMA",
                            stroke: ema26.stroke(),
                            windowSize: ema26.options().windowSize
                        },
                        {
                            yAccessor: ema12.accessor(),
                            type: "EMA",
                            stroke: ema12.stroke(),
                            windowSize: ema12.options().windowSize
                        }
                    ]}
                />

                <ZoomButtons />
                <OHLCTooltip textFill="white" origin={[8, 16]} />

            </Chart>

            <CrossHairCursor />
        </ChartCanvas>

        <div className="button-container" style={{ marginTop: "5px" }}>
            <GenerateNewBarButton
                onNewBarGenerated={handleNewBarGenerated}
                isFirstBarGenerated={isFirstBarGenerated}
                setIsFirstBarGenerated={setIsFirstBarGenerated}
            />
        </div>

    </div>  // ✅ Closing the outer div with margin
);
};

export default Simulation2;