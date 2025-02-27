use augurs::prophet::{
    Prophet, TrainingData, ProphetOptions, FeatureMode, 
    GrowthType, SeasonalityOption, wasmstan::WasmstanOptimizer, PredictionData
};
use csv::ReaderBuilder;
use chrono::NaiveDateTime;
use std::error::Error;
use plotters::prelude::*;

fn parse_datetime_to_timestamp(datetime_str: &str) -> Result<i64, Box<dyn Error>> {
    // Parse "2024-01-01 13:14" -> NaiveDateTime
    let naive_dt = NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M")?;
    // Convert to UNIX timestamp
    Ok(naive_dt.timestamp())
}

fn load_data_from_csv(file_path: &str) -> Result<(Vec<i64>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // Get `Start time` (column 1) and `Modified Count.Energy (Wh)` (column 7)
        if let (Some(ts_str), Some(energy_str)) = (record.get(1), record.get(7)) {
            // Convert timestamp to UNIX format
            if let (Ok(timestamp), Ok(energy)) = (
                parse_datetime_to_timestamp(ts_str.trim()),
                energy_str.trim().parse::<f64>(),
            ) {
                // Skip zero or negative energy values
                if energy > 0.0 {
                    timestamps.push(timestamp);
                    values.push(energy);
                }
            } else {
                println!("Skipping invalid row: {:?} -> {:?} | {:?}", ts_str, energy_str, record);
            }
        }
    }

    if timestamps.is_empty() || values.is_empty() {
        return Err("No valid data found in CSV. Please check file format.".into());
    }

    Ok((timestamps, values))
}

fn plot_forecast(timestamps: &Vec<i64>, future_timestamps: &Vec<i64>, actual_values: &Vec<f64>, predicted_values: &Vec<f64>) -> Result<(), Box<dyn Error>> {

    let output_file = "forecast.png";
    let root = BitMapBackend::new(output_file, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_x = *timestamps.first().unwrap();
    let max_x = future_timestamps.last().unwrap_or_else(|| timestamps.last().unwrap());
    let min_y = actual_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = actual_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("EV Charging Demand Forecast", ("Arial", 20))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..*max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    // Plot actual values (BLUE)
    chart.draw_series(LineSeries::new(
        timestamps.iter().zip(actual_values.iter()).map(|(x, y)| (*x, *y)),
        &BLUE,
    ))?
    .label("Actual Demand")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    // Plot predicted values (RED)
    chart.draw_series(LineSeries::new(
        timestamps.iter().zip(predicted_values.iter()).map(|(x, y)| (*x, *y)),
        &RED,
    ))?
    .label("Predicted Demand")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart.configure_series_labels().draw()?;

    println!("Forecast saved to {}", output_file);
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    // Load real data from CSV
    let (timestamps, values) = load_data_from_csv("data/site_data.csv")?;

    // Ensure we have enough data points
    if timestamps.len() < 30 {
        return Err("Not enough data points for forecasting. Try using more data.".into());
    }

    // Clone timestamps & values before passing them to TrainingData to avoid ownership issues
    let timestamps_clone = timestamps.clone();
    let values_clone = values.clone();

    // Create training data
    let data = TrainingData::new(timestamps_clone, values_clone)?;

    // Use WasmstanOptimizer
    let optimizer = WasmstanOptimizer::new();
    // Configure Prophet for volatile EV charging demand
    let options = ProphetOptions {
        // Linear growth model (captures increasing or decreasing trends)
        growth: GrowthType::Linear,
        
        // Multiplicative seasonality (captures large fluctuations in demand)
        seasonality_mode: FeatureMode::Multiplicative,

        // Hourly data: Enable strong daily patterns
        daily_seasonality: SeasonalityOption::Manual(true),

        // Enable weekly seasonality (weekdays vs. weekends)
        weekly_seasonality: SeasonalityOption::Manual(true),

        // Disable yearly seasonality (EV charging demand doesn't follow strict yearly cycles)
        yearly_seasonality: SeasonalityOption::Manual(false),

        ..Default::default()
    };

    // Initialize Prophet with optimized settings
    let mut prophet = Prophet::new(options, optimizer);

    // Fit the model
    prophet.fit(data, Default::default())?;

    // Find last timestamp in dataset
    let last_timestamp = *timestamps.last().unwrap();

    // Generate timestamps for next 7 days (168 hours)
    let future_timestamps: Vec<i64> = (1..=168).map(|i| last_timestamp + i * 3600).collect();

    // Convert `future_timestamps` into `PredictionData`
    let future_data = PredictionData::new(future_timestamps.clone());
    let predictions = prophet.predict(Some(future_data))?;

    // Predict future demand
    //let future_horizon = 48; // Forecast next 48 hours
    //let predictions = prophet.predict(Some(future_horizon))?;
    //let predictions = prophet.predict(None)?;

    // Print predictions with timestamps
    println!("Timestamp | Predicted Demand");
    for (timestamp, prediction) in timestamps.iter().zip(predictions.yhat.point.iter()) {
        println!("{} | {}", timestamp, prediction);
    }

    // Uncomment if you want additional details
    // println!("Predictions: {:?}", predictions.yhat.point);
    // println!("Lower bounds: {:?}", predictions.yhat.lower.unwrap());
    // println!("Upper bounds: {:?}", predictions.yhat.upper.unwrap());
    
    // Extract predicted values
    let predicted_values = predictions.yhat.point.clone();

    // Call the function to generate the plot
    plot_forecast(&timestamps, &future_timestamps, &values, &predicted_values)?;


    Ok(())
}

