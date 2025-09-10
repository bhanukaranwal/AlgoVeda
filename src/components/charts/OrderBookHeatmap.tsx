import React, { useEffect, useRef, useState } from 'react';
import { Paper, Typography, Box, Switch, FormControlLabel } from '@mui/material';
import * as d3 from 'd3';

interface OrderBookData {
  price: number;
  bidSize: number;
  askSize: number;
  timestamp: number;
}

interface OrderBookHeatmapProps {
  symbol: string;
  height?: number;
  width?: number;
  levels?: number;
  animate?: boolean;
}

const OrderBookHeatmap: React.FC<OrderBookHeatmapProps> = ({
  symbol,
  height = 400,
  width = 600,
  levels = 20,
  animate = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<OrderBookData[]>([]);
  const [showAnimation, setShowAnimation] = useState(animate);

  useEffect(() => {
    // Simulate real-time order book data
    const interval = setInterval(() => {
      const newData = generateMockOrderBookData(levels);
      setData(newData);
    }, 100);

    return () => clearInterval(interval);
  }, [levels]);

  useEffect(() => {
    if (!data.length || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.price) as [number, number])
      .range([0, innerWidth]);

    const yScale = d3.scaleBand()
      .domain(data.map((_, i) => i.toString()))
      .range([0, innerHeight])
      .padding(0.1);

    const maxSize = d3.max(data, d => Math.max(d.bidSize, d.askSize)) || 1;

    const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain([0, maxSize]);

    // Bid bars (left side)
    g.selectAll(".bid-bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bid-bar")
      .attr("x", d => xScale(d.price) - (d.bidSize / maxSize) * (innerWidth / 4))
      .attr("y", (_, i) => yScale(i.toString()) || 0)
      .attr("width", d => (d.bidSize / maxSize) * (innerWidth / 4))
      .attr("height", yScale.bandwidth())
      .attr("fill", d => colorScale(d.bidSize))
      .attr("opacity", 0.8)
      .on("mouseover", function(event, d) {
        d3.select(this).attr("opacity", 1);
        
        // Tooltip
        const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.8)")
          .style("color", "white")
          .style("padding", "5px")
          .style("border-radius", "3px")
          .style("pointer-events", "none");

        tooltip.html(`Bid: ${d.bidSize} @ $${d.price.toFixed(4)}`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this).attr("opacity", 0.8);
        d3.selectAll(".tooltip").remove();
      });

    // Ask bars (right side)
    g.selectAll(".ask-bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "ask-bar")
      .attr("x", d => xScale(d.price))
      .attr("y", (_, i) => yScale(i.toString()) || 0)
      .attr("width", d => (d.askSize / maxSize) * (innerWidth / 4))
      .attr("height", yScale.bandwidth())
      .attr("fill", d => colorScale(d.askSize))
      .attr("opacity", 0.8)
      .on("mouseover", function(event, d) {
        d3.select(this).attr("opacity", 1);
        
        const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.8)")
          .style("color", "white")
          .style("padding", "5px")
          .style("border-radius", "3px")
          .style("pointer-events", "none");

        tooltip.html(`Ask: ${d.askSize} @ $${d.price.toFixed(4)}`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this).attr("opacity", 0.8);
        d3.selectAll(".tooltip").remove();
      });

    // Center line
    g.append("line")
      .attr("x1", innerWidth / 2)
      .attr("y1", 0)
      .attr("x2", innerWidth / 2)
      .attr("y2", innerHeight)
      .attr("stroke", "#333")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    // Axes
    const xAxis = d3.axisBottom(xScale).tickFormat(d3.format(".4f"));
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);

    // Labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (innerHeight / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Order Book Levels");

    g.append("text")
      .attr("transform", `translate(${innerWidth / 2}, ${innerHeight + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Price ($)");

    // Animation
    if (showAnimation) {
      g.selectAll("rect")
        .transition()
        .duration(100)
        .attr("opacity", 0.8);
    }

  }, [data, width, height, showAnimation]);

  const generateMockOrderBookData = (levels: number): OrderBookData[] => {
    const basePrice = 150 + Math.sin(Date.now() / 10000) * 5;
    const spread = 0.02;
    
    return Array.from({ length: levels }, (_, i) => {
      const offset = (i - levels / 2) * 0.01;
      const price = basePrice + offset;
      
      return {
        price,
        bidSize: Math.floor(Math.random() * 1000) + 100,
        askSize: Math.floor(Math.random() * 1000) + 100,
        timestamp: Date.now()
      };
    });
  };

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          {symbol} Order Book Heatmap
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showAnimation}
              onChange={(e) => setShowAnimation(e.target.checked)}
            />
          }
          label="Animation"
        />
      </Box>
      
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ border: '1px solid #ddd' }}
      />
      
      <Box mt={2} display="flex" justifyContent="space-between">
        <Typography variant="caption" color="primary">
          ← Bids
        </Typography>
        <Typography variant="caption" color="error">
          Asks →
        </Typography>
      </Box>
    </Paper>
  );
};

export default OrderBookHeatmap;
