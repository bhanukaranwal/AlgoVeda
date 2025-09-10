/*!
 * WebAssembly Visualization Module
 * High-performance chart rendering and data visualization for web
 */

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use js_sys::{Array, Object, Uint8ClampedArray};
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChartRenderer {
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    width: u32,
    height: u32,
    data_points: Vec<DataPoint>,
    chart_config: ChartConfig,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DataPoint {
    x: f64,
    y: f64,
    timestamp: f64,
    value: f64,
    label: String,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChartConfig {
    chart_type: ChartType,
    show_grid: bool,
    show_axes: bool,
    animate: bool,
    color_scheme: String,
    margin_top: u32,
    margin_bottom: u32,
    margin_left: u32,
    margin_right: u32,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub enum ChartType {
    Line,
    Area,
    Bar,
    Candlestick,
    Scatter,
    Heatmap,
    Surface3D,
}

#[wasm_bindgen]
impl ChartRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<ChartRenderer, JsValue> {
        let context = canvas
            .get_context("2d")?
            .ok_or("Failed to get 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let width = canvas.width();
        let height = canvas.height();

        Ok(ChartRenderer {
            canvas,
            context,
            width,
            height,
            data_points: Vec::new(),
            chart_config: ChartConfig::default(),
        })
    }

    #[wasm_bindgen]
    pub fn set_data(&mut self, data: JsValue) -> Result<(), JsValue> {
        let data_array: Array = data.into();
        self.data_points.clear();

        for i in 0..data_array.length() {
            let point_obj = data_array.get(i);
            let point = self.parse_data_point(point_obj)?;
            self.data_points.push(point);
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn render_line_chart(&self) -> Result<(), JsValue> {
        self.clear_canvas();
        
        if self.data_points.is_empty() {
            return Ok(());
        }

        // Calculate scales
        let (x_min, x_max, y_min, y_max) = self.calculate_bounds();
        let (x_scale, y_scale) = self.calculate_scales(x_min, x_max, y_min, y_max);

        // Draw grid if enabled
        if self.chart_config.show_grid {
            self.draw_grid(&x_scale, &y_scale)?;
        }

        // Draw axes if enabled
        if self.chart_config.show_axes {
            self.draw_axes(&x_scale, &y_scale)?;
        }

        // Draw line
        self.draw_line(&x_scale, &y_scale)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn render_candlestick_chart(&self, ohlc_data: JsValue) -> Result<(), JsValue> {
        self.clear_canvas();
        
        let ohlc_array: Array = ohlc_data.into();
        if ohlc_array.length() == 0 {
            return Ok(());
        }

        // Parse OHLC data
        let mut ohlc_points = Vec::new();
        for i in 0..ohlc_array.length() {
            let candle_obj = ohlc_array.get(i);
            let candle = self.parse_ohlc_point(candle_obj)?;
            ohlc_points.push(candle);
        }

        // Calculate bounds for OHLC data
        let (x_min, x_max, y_min, y_max) = self.calculate_ohlc_bounds(&ohlc_points);
        let (x_scale, y_scale) = self.calculate_scales(x_min, x_max, y_min, y_max);

        // Draw grid and axes
        if self.chart_config.show_grid {
            self.draw_grid(&x_scale, &y_scale)?;
        }
        if self.chart_config.show_axes {
            self.draw_axes(&x_scale, &y_scale)?;
        }

        // Draw candlesticks
        self.draw_candlesticks(&ohlc_points, &x_scale, &y_scale)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn render_heatmap(&self, matrix_data: JsValue) -> Result<(), JsValue> {
        self.clear_canvas();
        
        let matrix: Array = matrix_data.into();
        if matrix.length() == 0 {
            return Ok(());
        }

        let rows = matrix.length() as usize;
        let cols = if rows > 0 {
            let first_row: Array = matrix.get(0).into();
            first_row.length() as usize
        } else {
            0
        };

        if cols == 0 {
            return Ok(());
        }

        // Parse matrix data
        let mut data_matrix = vec![vec![0.0; cols]; rows];
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for i in 0..rows {
            let row: Array = matrix.get(i as u32).into();
            for j in 0..cols.min(row.length() as usize) {
                let value = row.get(j as u32).as_f64().unwrap_or(0.0);
                data_matrix[i][j] = value;
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }

        // Render heatmap
        self.draw_heatmap(&data_matrix, min_val, max_val)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn render_real_time(&mut self, new_data_point: JsValue) -> Result<(), JsValue> {
        // Add new data point
        let point = self.parse_data_point(new_data_point)?;
        self.data_points.push(point);

        // Keep only last N points for performance
        const MAX_POINTS: usize = 1000;
        if self.data_points.len() > MAX_POINTS {
            self.data_points.drain(0..self.data_points.len() - MAX_POINTS);
        }

        // Re-render chart
        self.render_line_chart()
    }

    // Private helper methods
    fn parse_data_point(&self, point_obj: JsValue) -> Result<DataPoint, JsValue> {
        let obj: Object = point_obj.into();
        
        let x = js_sys::Reflect::get(&obj, &"x".into())?
            .as_f64().unwrap_or(0.0);
        let y = js_sys::Reflect::get(&obj, &"y".into())?
            .as_f64().unwrap_or(0.0);
        let timestamp = js_sys::Reflect::get(&obj, &"timestamp".into())?
            .as_f64().unwrap_or(0.0);
        let value = js_sys::Reflect::get(&obj, &"value".into())?
            .as_f64().unwrap_or(0.0);
        let label = js_sys::Reflect::get(&obj, &"label".into())?
            .as_string().unwrap_or_default();

        Ok(DataPoint {
            x, y, timestamp, value, label,
        })
    }

    fn parse_ohlc_point(&self, candle_obj: JsValue) -> Result<OHLCPoint, JsValue> {
        let obj: Object = candle_obj.into();
        
        let timestamp = js_sys::Reflect::get(&obj, &"timestamp".into())?
            .as_f64().unwrap_or(0.0);
        let open = js_sys::Reflect::get(&obj, &"open".into())?
            .as_f64().unwrap_or(0.0);
        let high = js_sys::Reflect::get(&obj, &"high".into())?
            .as_f64().unwrap_or(0.0);
        let low = js_sys::Reflect::get(&obj, &"low".into())?
            .as_f64().unwrap_or(0.0);
        let close = js_sys::Reflect::get(&obj, &"close".into())?
            .as_f64().unwrap_or(0.0);
        let volume = js_sys::Reflect::get(&obj, &"volume".into())?
            .as_f64().unwrap_or(0.0);

        Ok(OHLCPoint {
            timestamp, open, high, low, close, volume,
        })
    }

    fn clear_canvas(&self) {
        self.context.clear_rect(0.0, 0.0, self.width as f64, self.height as f64);
        self.context.set_fill_style(&"#1e1e1e".into());
        self.context.fill_rect(0.0, 0.0, self.width as f64, self.height as f64);
    }

    fn calculate_bounds(&self) -> (f64, f64, f64, f64) {
        if self.data_points.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let x_min = self.data_points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x_max = self.data_points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.data_points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y_max = self.data_points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        (x_min, x_max, y_min, y_max)
    }

    fn calculate_ohlc_bounds(&self, ohlc_points: &[OHLCPoint]) -> (f64, f64, f64, f64) {
        if ohlc_points.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let x_min = ohlc_points.iter().map(|p| p.timestamp).fold(f64::INFINITY, f64::min);
        let x_max = ohlc_points.iter().map(|p| p.timestamp).fold(f64::NEG_INFINITY, f64::max);
        let y_min = ohlc_points.iter().map(|p| p.low).fold(f64::INFINITY, f64::min);
        let y_max = ohlc_points.iter().map(|p| p.high).fold(f64::NEG_INFINITY, f64::max);

        (x_min, x_max, y_min, y_max)
    }

    fn calculate_scales(&self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> (Scale, Scale) {
        let plot_width = self.width as f64 - self.chart_config.margin_left as f64 - self.chart_config.margin_right as f64;
        let plot_height = self.height as f64 - self.chart_config.margin_top as f64 - self.chart_config.margin_bottom as f64;

        let x_scale = Scale {
            domain_min: x_min,
            domain_max: x_max,
            range_min: self.chart_config.margin_left as f64,
            range_max: self.chart_config.margin_left as f64 + plot_width,
        };

        let y_scale = Scale {
            domain_min: y_min,
            domain_max: y_max,
            range_min: self.chart_config.margin_top as f64 + plot_height,
            range_max: self.chart_config.margin_top as f64,
        };

        (x_scale, y_scale)
    }

    fn draw_grid(&self, x_scale: &Scale, y_scale: &Scale) -> Result<(), JsValue> {
        self.context.set_stroke_style(&"rgba(255, 255, 255, 0.1)".into());
        self.context.set_line_width(1.0);

        // Vertical grid lines
        for i in 0..=10 {
            let x = x_scale.range_min + (x_scale.range_max - x_scale.range_min) * i as f64 / 10.0;
            self.context.begin_path();
            self.context.move_to(x, y_scale.range_max);
            self.context.line_to(x, y_scale.range_min);
            self.context.stroke();
        }

        // Horizontal grid lines
        for i in 0..=10 {
            let y = y_scale.range_max + (y_scale.range_min - y_scale.range_max) * i as f64 / 10.0;
            self.context.begin_path();
            self.context.move_to(x_scale.range_min, y);
            self.context.line_to(x_scale.range_max, y);
            self.context.stroke();
        }

        Ok(())
    }

    fn draw_axes(&self, x_scale: &Scale, y_scale: &Scale) -> Result<(), JsValue> {
        self.context.set_stroke_style(&"rgba(255, 255, 255, 0.8)".into());
        self.context.set_line_width(2.0);

        // X-axis
        self.context.begin_path();
        self.context.move_to(x_scale.range_min, y_scale.range_min);
        self.context.line_to(x_scale.range_max, y_scale.range_min);
        self.context.stroke();

        // Y-axis
        self.context.begin_path();
        self.context.move_to(x_scale.range_min, y_scale.range_min);
        self.context.line_to(x_scale.range_min, y_scale.range_max);
        self.context.stroke();

        Ok(())
    }

    fn draw_line(&self, x_scale: &Scale, y_scale: &Scale) -> Result<(), JsValue> {
        if self.data_points.len() < 2 {
            return Ok(());
        }

        self.context.set_stroke_style(&"#2196f3".into());
        self.context.set_line_width(2.0);
        self.context.begin_path();

        let first_point = &self.data_points[0];
        let first_x = x_scale.scale(first_point.x);
        let first_y = y_scale.scale(first_point.y);
        self.context.move_to(first_x, first_y);

        for point in &self.data_points[1..] {
            let x = x_scale.scale(point.x);
            let y = y_scale.scale(point.y);
            self.context.line_to(x, y);
        }

        self.context.stroke();
        Ok(())
    }

    fn draw_candlesticks(&self, ohlc_points: &[OHLCPoint], x_scale: &Scale, y_scale: &Scale) -> Result<(), JsValue> {
        let candle_width = (x_scale.range_max - x_scale.range_min) / ohlc_points.len() as f64 * 0.8;

        for (i, candle) in ohlc_points.iter().enumerate() {
            let x = x_scale.range_min + (x_scale.range_max - x_scale.range_min) * i as f64 / ohlc_points.len() as f64;
            let open_y = y_scale.scale(candle.open);
            let high_y = y_scale.scale(candle.high);
            let low_y = y_scale.scale(candle.low);
            let close_y = y_scale.scale(candle.close);

            let is_bullish = candle.close > candle.open;
            let color = if is_bullish { "#4caf50" } else { "#f44336" };

            // Draw high-low line
            self.context.set_stroke_style(&"#ffffff".into());
            self.context.set_line_width(1.0);
            self.context.begin_path();
            self.context.move_to(x, high_y);
            self.context.line_to(x, low_y);
            self.context.stroke();

            // Draw body
            self.context.set_fill_style(&color.into());
            let body_top = if is_bullish { close_y } else { open_y };
            let body_height = (close_y - open_y).abs();
            self.context.fill_rect(x - candle_width / 2.0, body_top, candle_width, body_height);
        }

        Ok(())
    }

    fn draw_heatmap(&self, data_matrix: &[Vec<f64>], min_val: f64, max_val: f64) -> Result<(), JsValue> {
        let rows = data_matrix.len();
        let cols = if rows > 0 { data_matrix[0].len() } else { 0 };

        let cell_width = self.width as f64 / cols as f64;
        let cell_height = self.height as f64 / rows as f64;

        for (i, row) in data_matrix.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let normalized_value = if max_val > min_val {
                    (value - min_val) / (max_val - min_val)
                } else {
                    0.5
                };

                let color = self.value_to_color(normalized_value);
                self.context.set_fill_style(&color.into());
                
                let x = j as f64 * cell_width;
                let y = i as f64 * cell_height;
                self.context.fill_rect(x, y, cell_width, cell_height);
            }
        }

        Ok(())
    }

    fn value_to_color(&self, normalized_value: f64) -> String {
        // Convert normalized value (0-1) to color (blue to red)
        let red = (normalized_value * 255.0) as u8;
        let blue = ((1.0 - normalized_value) * 255.0) as u8;
        format!("rgb({}, 0, {})", red, blue)
    }
}

#[derive(Debug, Clone)]
struct Scale {
    domain_min: f64,
    domain_max: f64,
    range_min: f64,
    range_max: f64,
}

impl Scale {
    fn scale(&self, value: f64) -> f64 {
        let normalized = if self.domain_max > self.domain_min {
            (value - self.domain_min) / (self.domain_max - self.domain_min)
        } else {
            0.5
        };
        
        self.range_min + normalized * (self.range_max - self.range_min)
    }
}

#[derive(Debug, Clone)]
struct OHLCPoint {
    timestamp: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            chart_type: ChartType::Line,
            show_grid: true,
            show_axes: true,
            animate: false,
            color_scheme: "default".to_string(),
            margin_top: 20,
            margin_bottom: 40,
            margin_left: 60,
            margin_right: 20,
        }
    }
}

// Export types for JavaScript
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "ChartRenderer")]
    pub type ChartRendererType;
}
