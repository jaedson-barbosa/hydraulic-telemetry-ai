use rand::distributions::Uniform;
use rand::{seq::SliceRandom, Rng};
use rand::thread_rng;
use std::{fs::File, io::BufReader};
use neuroflow::{ActivatorType::Tanh, FeedForward};

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
        let mut a = [last.adc_state.battery_mv, self.adc_state.battery_mv, self.adc_state.ldo_inp_mv, 0].map(|v| Self::norm(v));
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

fn load_data(path: &str) -> Vec<[f32;4]> {
    let data_file = File::open(path).unwrap();
        let file_reader = BufReader::new(data_file);
        let data: Vec<DeviceState> = serde_json::from_reader(file_reader).unwrap();
        let len = data.len() as f32;
        data.windows(2)
            .enumerate()
            .map(|(index, item)| {
                let percentage = (len - index as f32) / len;
                item[1].get_ai_data(item[0], percentage)
            })
            .collect::<Vec<[f32; 4]>>()
}

fn main() {
    let mut data = load_data("data/data1.json"); // rand::random::<f32>()
    
    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    let split_pos = data.len() * 8 / 10;
    let (train_data, test_data) = data.split_at_mut(split_pos);

    let rand = || -> (&[f32], &[f32]){
        let mut rnd_range = thread_rng();
        let i = rnd_range.sample(Uniform::new(0, train_data.len()));
        (&train_data[i][..3], &train_data[i][3..])
    };

    let mut nn = FeedForward::new(&[3, 3, 3, 1], || rand::random::<f32>());
    for i in 0..10_000_000 {
        let item = train_data[i % train_data.len()];
        nn.fit(&item[..3], &item[3..]);
    }

    let mut writer = csv::Writer::from_path("out/result.csv").unwrap();
    let mut total_abs_error = 0.0;
    let mut total_squared_error = 0.0;
    for item in test_data.iter() {
        let input = &item[..3];
        let output = nn.calc(input)[0];
        let expected = item[3];
        println!("for {input:?}, {expected} -> {output}");

        let difference = expected - output;
        total_abs_error += difference.abs();
        total_squared_error += difference * difference;

        let test = Test {
            battery_mv: (input[0] * 800.0 + 3300.0) as u16,
            percentage: (expected * 1000.0) as u16,
            prediction: (output * 1000.0) as u16,
        };
        writer.serialize(test).unwrap();
        writer.flush().unwrap()
    }
    let len = test_data.len() as f32;
    let mae = total_abs_error / len;
    let rmse = (total_squared_error / len).sqrt();
    println!("Mean absolute error: {mae:.4}");
    println!("Root Mean Square Error: {rmse:.4}");

    let str = postcard::to_vec::<_, 1024>(&nn).unwrap();
    std::fs::write("model/model", &str).unwrap();
}
