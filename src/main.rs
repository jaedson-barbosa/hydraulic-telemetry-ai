use neuroflow::FeedForward;
use plotters::prelude::{ChartBuilder, IntoDrawingArea, PathElement, SeriesLabelPosition};
use plotters::series::LineSeries;
use plotters::style::{BLACK, BLUE, RED, WHITE};
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::cmp::Ordering;
use std::{fs::File, io::BufReader};

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default)]
pub struct ADCRead {
    pub battery_mv: u16,
    pub ldo_inp_mv: u16,
    pub pressure_mv: u16,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default)]
pub struct DeviceState {
    pub adc_state: ADCRead,
    pub n_pulses: u16,
    pub time_sec: u64,
}

impl DeviceState {
    pub fn get_ai_data(&self, last: Self, percentage: f32) -> [f32; 4] {
        let mut a = [
            last.adc_state.battery_mv,
            self.adc_state.battery_mv,
            self.adc_state.ldo_inp_mv,
            0,
        ]
        .map(|v| Self::norm(v));
        a[3] = percentage;
        a
    }

    fn norm(value: u16) -> f32 {
        (value - 3300) as f32 / 800.0
    }
}

#[derive(serde::Serialize, Debug)]
struct Test {
    battery_mv: u16,
    percentage: u16,
    prediction: u16,
}

fn load_data(path: &str) -> (Vec<DeviceState>, Vec<[f32; 4]>) {
    let data_file = File::open(path).unwrap();
    let file_reader = BufReader::new(data_file);
    let data: Vec<DeviceState> = serde_json::from_reader(file_reader).unwrap();
    let max = data.last().unwrap().time_sec as f32;
    let input_data = data
        .windows(2)
        .map(|item| {
            let percentage = (max - item[1].time_sec as f32) / max;
            item[1].get_ai_data(item[0], percentage)
        })
        .collect();
    (data, input_data)
}

fn main() {
    let (source_data, mut data) = load_data("data/data1.json"); // rand::random::<f32>()

    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    let train_len = data.len() / 20;
    let start = std::time::Instant::now();
    let mut nn = FeedForward::new(&[3, 3, 3, 1], || rand::random::<f32>());
    nn.learn_rate = 0.5;
    for i in 0..100_000 {
        let item = data[i % train_len];
        nn.fit(&item[..3], &[item[3]])
    }
    let end = std::time::Instant::now();
    let elapsed = (end - start).as_millis();
    println!("Training time: {elapsed} ms");

    let mut writer = csv::Writer::from_path("out/result.csv").unwrap();
    let mut total_abs_error = 0.0;
    let mut total_squared_error = 0.0;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut y_nn = Vec::new();
    data.sort_by(|a, b| {
        if a[3] < b[3] {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    });
    for (i, item) in data.iter().enumerate() {
        let input = &item[..3];
        let output = nn.calc(input)[0];
        let expected = item[3];

        let difference = expected - output;
        total_abs_error += difference.abs();
        total_squared_error += difference * difference;

        let test = Test {
            battery_mv: (input[0] * 800.0 + 3300.0) as u16,
            percentage: (expected * 1000.0) as u16,
            prediction: (output * 1000.0) as u16,
        };
        writer.serialize(test).unwrap();
        writer.flush().unwrap();

        x.push((source_data[i].time_sec as f32) / 3600.0);
        y.push(expected);
        y_nn.push(output);
    }
    let len = data.len() as f32;
    let mae = total_abs_error / len;
    let rmse = (total_squared_error / len).sqrt();
    println!("Mean absolute error: {mae:.4}");
    println!("Root Mean Square Error: {rmse:.4}");

    let str = postcard::to_vec::<_, 1024>(&nn).unwrap();
    std::fs::write("model/model", &str).unwrap();

    let root = plotters_svg::SVGBackend::new("soc1.svg", (550, 360)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_end = (source_data.last().unwrap().time_sec as f32) / 3600.0;
    let min_time = (source_data[0].time_sec as f32) / 3600.0;
    let min_bat_v = (source_data.last().unwrap().adc_state.battery_mv as f32) / 1000.0;
    let max_bat_v = (source_data[0].adc_state.battery_mv as f32) / 1000.0;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .set_left_and_bottom_label_area_size(40)
        .right_y_label_area_size(40)
        .build_cartesian_2d(min_time..x_end, 0.0..100.0f32)
        .unwrap()
        .set_secondary_coord(min_time..x_end, min_bat_v..max_bat_v);
    chart
        .configure_mesh()
        .x_desc("Tempo (h)")
        .y_desc("SOC (%)")
        .draw()
        .unwrap();
    chart
        .configure_secondary_axes()
        .y_desc("Tensão (V)")
        .draw()
        .unwrap();
    let values = x.iter().zip(y.iter()).map(|v| (*v.0, *v.1 * 100.0));
    chart
        .draw_series(LineSeries::new(values, &RED))
        .unwrap()
        .label("SOC correto")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    let values = x
        .iter()
        .zip(y_nn.iter())
        .map(|v| (*v.0, *v.1 * 100.0));
    chart
        .draw_series(LineSeries::new(values, &BLUE))
        .unwrap()
        .label("SOC estimado")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    let values = x
        .iter()
        .zip(source_data.iter().map(|v| v.adc_state.battery_mv as f32))
        .map(|v| (*v.0, v.1 / 1000.0));
    chart
        .draw_secondary_series(LineSeries::new(values, &BLACK))
        .unwrap()
        .label("Tensão bat.")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()
        .unwrap();
    root.present().unwrap();
}
