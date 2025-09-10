/*!
 * WebAssembly Charts Module
 * High-performance chart rendering for trading applications
 */

use wasm_bindgen::prelude::*;
use web_sys::{
    CanvasRenderingContext2d, 
    HtmlCanvasElement, 
    ImageData, 
    Path2d
};
use js_sys::{Array, Object, Uint8ClampedArray};

#[wasm_bindgen]
pub struct ChartsModule {
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl ChartsModule {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<ChartsModule, JsValue> {
        let context = canvas
            .get_context("2d")?
            .ok_or("Failed to get 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let width = canvas.width();
        let height = canvas.height();

        Ok(ChartsModule {
            canvas,
            context,
            width,
            height,
        })
    }

    #[wasm_bindgen]
    pub fn clear_canvas(&self) {
        self.context.clear_rect(0.0, 0.0, self.width as f64, self.height as f64);
        self.context.set_fill_style(&"#1e1e1e".into());
        self.context.fill_rect(0.0, 0.0, self.width as f64, self.height as f64);
    }

    #[wasm_bindgen]
    pub fn draw_line_chart(&self, data: JsValue, options: JsValue) -> Result<(), JsValue> {
        self.clear_canvas();
        
        let data_array: Array = data.into();
        if data_array.length() == 0 {
            return Ok(());
        }

        let parsed_data = self.parse_chart_data(&data_array)?;
        let chart_options = self.parse_chart_options(options)?;

        // Calculate bounds
        let (x_min, x_max, y_min, y_max) = self.calculate_data_bounds(&parsed_data);
        
        // Draw axes and grid
        self.draw_chart_frame(x_min, x_max, y_min, y_max, &chart_options)?;
        
        // Draw line
        self.draw_line_series(&parsed_data, x_min, x_max, y_min, y_max, "#2196f3")?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_candlestick_chart(&self, data: JsValue, options: JsValue) -> Result<(), JsValue> {
        self.clear_canvas();
        
        let data_array: Array = data.into();
        if data_array.length() == 0 {
            return Ok(());
        }

        let ohlc_data = self.parse_ohlc_data(&data_array)?;
        let chart_options = self.parse_chart_options(options)?;

        // Calculate bounds
        let (x_min, x_max, y_min, y_max) = self.calculate_ohlc_bounds(&ohlc_data);
        
        // Draw axes and grid
        self.draw_chart_frame(x_min, x_max, y_min, y_max, &chart_options)?;
        
        // Draw candlesticks
        self.draw_candlesticks(&ohlc_data, x_min, x_max, y_min, y_max)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_volume_bars(&self, data: JsValue, y_offset: f64) -> Result<(), JsValue> {
        let data_array: Array = data.into();
        if data_array.length() == 0 {
            return Ok(());
        }

        let volume_data = self.parse_volume_data(&data_array)?;
        let max_volume = volume_data.iter().map(|v| v.volume).fold(0.0, f64::max);

        let bar_width = (self.width as f64) / (volume_data.len() as f64 * 1.2);
        let max_bar_height = 100.0; // pixels

        self.context.set_fill_style(&"rgba(100, 149, 237, 0.5)".into());

        for (i, volume_point) in volume_data.iter().enumerate() {
            let x = (i as f64 + 0.1) * bar_width * 1.2;
            let normalized_volume = volume_point.volume / max_volume;
            let bar_height = normalized_volume * max_bar_height;
            
            self.context.fill_rect(
                x, 
                y_offset - bar_height, 
                bar_width, 
                bar_height
            );
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_technical_indicators(&self, sma_data: JsValue, ema_data: JsValue, bounds: JsValue) -> Result<(), JsValue> {
        let bounds_obj: Object = bounds.into();
        let x_min = js_sys::Reflect::get(&bounds_obj, &"x_min".into())?
            .as_f64().unwrap_or(0.0);
        let x_max = js_sys::Reflect::get(&bounds_obj, &"x_max".into())?
            .as_f64().unwrap_or(100.0);
        let y_min = js_sys::Reflect::get(&bounds_obj, &"y_min".into())?
            .as_f64().unwrap_or(0.0);
        let y_max = js_sys::Reflect::get(&bounds_obj, &"y_max".into())?
            .as_f64().unwrap_or(100.0);

        // Draw SMA
        let sma_array: Array = sma_data.into();
        if sma_array.length() > 0 {
            let sma_points = self.parse_chart_data(&sma_array)?;
            self.draw_line_series(&sma_points, x_min, x_max, y_min, y_max, "#ff9800")?;
        }

        // Draw EMA
        let ema_array: Array = ema_data.into();
        if ema_array.length() > 0 {
            let ema_points = self.parse_chart_data(&ema_array)?;
            self.draw_line_series(&ema_points, x_min, x_max, y_min, y_max, "#4caf50")?;
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_heatmap(&self, matrix_data: JsValue, width_cells: u32, height_cells: u32) -> Result<(), JsValue> {
        self.clear_canvas();
        
        let matrix: Array = matrix_data.into();
        let cell_width = self.width as f64 / width_cells as f64;
        let cell_height = self.height as f64 / height_cells as f64;

        // Find min/max values for color scaling
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        let mut data_matrix = Vec::new();
        
        for i in 0..height_cells {
            let mut row = Vec::new();
            for j in 0..width_cells {
                let index = (i * width_cells + j) as usize;
                let value = if index < matrix.length() as usize {
                    matrix.get(index as u32).as_f64().unwrap_or(0.0)
                } else {
                    0.0
                };
                
                row.push(value);
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
            data_matrix.push(row);
        }

        // Draw heatmap cells
        for (i, row) in data_matrix.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let normalized_value = if max_val > min_val {
                    (value - min_val) / (max_val - min_val)
                } else {
                    0.5
                };

                let color = self.value_to_heatmap_color(normalized_value);
                self.context.set_fill_style(&color.into());
                
                let x = j as f64 * cell_width;
                let y = i as f64 * cell_height;
                
                self.context.fill_rect(x, y, cell_width, cell_height);
            }
        }

        Ok(())
    }

    // Private helper methods

    fn parse_chart_data(&self, data_array: &Array) -> Result<Vec<ChartPoint>, JsValue> {
        let mut points = Vec::new();
        
        for i in 0..data_array.length() {
            let point_obj: Object = data_array.get(i).into();
            let x = js_sys::Reflect::get(&point_obj, &"x".into())?
                .as_f64().unwrap_or(0.0);
            let y = js_sys::Reflect::get(&point_obj, &"y".into())?
                .as_f64().unwrap_or(0.0);
            
            points.push(ChartPoint { x, y });
        }
        
        Ok(points)
    }

    fn parse_ohlc_data(&self, data_array: &Array) -> Result<Vec<OHLCPoint>, JsValue> {
        let mut points = Vec::new();
        
        for i in 0..data_array.length() {
            let point_obj: Object = data_array.get(i).into();
            let timestamp = js_sys::Reflect::get(&point_obj, &"timestamp".into())?
                .as_f64().unwrap_or(0.0);
            let open = js_sys::Reflect::get(&point_obj, &"open".into())?
                .as_f64().unwrap_or(0.0);
            let high = js_sys::Reflect::get(&point_obj, &"high".into())?
                .as_f64().unwrap_or(0.0);
            let low = js_sys::Reflect::get(&point_obj, &"low".into())?
                .as_f64().unwrap_or(0.0);
            let close = js_sys::Reflect::get(&point_obj, &"close".into())?
                .as_f64().unwrap_or(0.0);
            
            points.push(OHLCPoint {
                timestamp, open, high, low, close,
            });
        }
        
        Ok(points)
    }

    fn parse_volume_data(&self, data_array: &Array) -> Result<Vec<VolumePoint>, JsValue> {
        let mut points = Vec::new();
        
        for i in 0..data_array.length() {
            let point_obj: Object = data_array.get(i).into();
            let timestamp = js_sys::Reflect::get(&point_obj, &"timestamp".into())?
                .as_f64().unwrap_or(0.0);
            let volume = js_sys::Reflect::get(&point_obj, &"volume".into())?
                .as_f64().unwrap_or(0.0);
            
            points.push(VolumePoint { timestamp, volume });
        }
        
        Ok(points)
    }

    fn parse_chart_options(&self, options: JsValue) -> Result<ChartOptions, JsValue> {
        let options_obj: Object = options.into();
        
        Ok(ChartOptions {
            show_grid: js_sys::Reflect::get(&options_obj, &"showGrid".into())?
                .as_bool().unwrap_or(true),
            show_axes: js_sys::Reflect::get(&options_obj, &"showAxes".into())?
                .as_bool().unwrap_or(true),
            margin: 50.0,
        })
    }

    fn calculate_data_bounds(&self, data: &[ChartPoint]) -> (f64, f64, f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let x_min = data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x_max = data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        (x_min, x_max, y_min, y_max)
    }

    fn calculate_ohlc_bounds(&self, data: &[OHLCPoint]) -> (f64, f64, f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let x_min = data.iter().map(|p| p.timestamp).fold(f64::INFINITY, f64::min);
        let x_max = data.iter().map(|p| p.timestamp).fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|p| p.low).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|p| p.high).fold(f64::NEG_INFINITY, f64::max);

        (x_min, x_max, y_min, y_max)
    }

    fn draw_chart_frame(&self, x_min: f64, x_max: f64, y_min: f64, y_max: f64, options: &ChartOptions) -> Result<(), JsValue> {
        let margin = options.margin;
        let plot_width = self.width as f64 - 2.0 * margin;
        let plot_height = self.height as f64 - 2.0 * margin;

        if options.show_grid {
            self.context.set_stroke_style(&"rgba(255, 255, 255, 0.1)".into());
            self.context.set_line_width(1.0);

            // Vertical grid lines
            for i in 0..=10 {
                let x = margin + plot_width * i as f64 / 10.0;
                self.context.begin_path();
                self.context.move_to(x, margin);
                self.context.line_to(x, margin + plot_height);
                self.context.stroke();
            }

            // Horizontal grid lines
            for i in 0..=10 {
                let y = margin + plot_height * i as f64 / 10.0;
                self.context.begin_path();
                self.context.move_to(margin, y);
                self.context.line_to(margin + plot_width, y);
                self.context.stroke();
            }
        }

        if options.show_axes {
            self.context.set_stroke_style(&"rgba(255, 255, 255, 0.8)".into());
            self.context.set_line_width(2.0);

            // X-axis
            self.context.begin_path();
            self.context.move_to(margin, margin + plot_height);
            self.context.line_to(margin + plot_width, margin + plot_height);
            self.context.stroke();

            // Y-axis
            self.context.begin_path();
            self.context.move_to(margin, margin);
            self.context.line_to(margin, margin + plot_height);
            self.context.stroke();
        }

        Ok(())
    }

    fn draw_line_series(&self, data: &[ChartPoint], x_min: f64, x_max: f64, y_min: f64, y_max: f64, color: &str) -> Result<(), JsValue> {
        if data.len() < 2 {
            return Ok(());
        }

        let margin = 50.0;
        let plot_width = self.width as f64 - 2.0 * margin;
        let plot_height = self.height as f64 - 2.0 * margin;

        self.context.set_stroke_style(&color.into());
        self.context.set_line_width(2.0);
        self.context.begin_path();

        let first_point = &data[0];
        let first_x = margin + ((first_point.x - x_min) / (x_max - x_min)) * plot_width;
        let first_y = margin + plot_height - ((first_point.y - y_min) / (y_max - y_min)) * plot_height;
        self.context.move_to(first_x, first_y);

        for point in &data[1..] {
            let x = margin + ((point.x - x_min) / (x_max - x_min)) * plot_width;
            let y = margin + plot_height - ((point.y - y_min) / (y_max - y_min)) * plot_height;
            self.context.line_to(x, y);
        }

        self.context.stroke();
        Ok(())
    }

    fn draw_candlesticks(&self, data: &[OHLCPoint], x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Result<(), JsValue> {
        let margin = 50.0;
        let plot_width = self.width as f64 - 2.0 * margin;
        let plot_height = self.height as f64 - 2.0 * margin;
        let candle_width = plot_width / data.len() as f64 * 0.8;

        for (i, candle) in data.iter().enumerate() {
            let x = margin + (i as f64 + 0.5) * plot_width / data.len() as f64;
            
            let open_y = margin + plot_height - ((candle.open - y_min) / (y_max - y_min)) * plot_height;
            let high_y = margin + plot_height - ((candle.high - y_min) / (y_max - y_min)) * plot_height;
            let low_y = margin + plot_height - ((candle.low - y_min) / (y_max - y_min)) * plot_height;
            let close_y = margin + plot_height - ((candle.close - y_min) / (y_max - y_min)) * plot_height;

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

    fn value_to_heatmap_color(&self, normalized_value: f64) -> String {
        // Convert normalized value (0-1) to color (blue to red)
        let red = (normalized_value * 255.0) as u8;
        let blue = ((1.0 - normalized_value) * 255.0) as u8;
        format!("rgb({}, 0, {})", red, blue)
    }
}

// Helper structs
#[derive(Debug, Clone)]
struct ChartPoint {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone)]
struct OHLCPoint {
    timestamp: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Debug, Clone)]
struct VolumePoint {
    timestamp: f64,
    volume: f64,
}

#[derive(Debug, Clone)]
struct ChartOptions {
    show_grid: bool,
    show_axes: bool,
    margin: f64,
}
