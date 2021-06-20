use crate::simulation::{SimulationSettings, SimulationSize, InitSettings};
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs::File;
use anyhow::Result;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Config {
    pub size: SimulationSize,
    pub init: InitSettings,
    pub step: SimulationSettings,
    pub control: Control,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Control {
    pub steps_per_frame: u32,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            steps_per_frame: 1,
        }
    }
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_yaml::from_reader(File::open(path)?)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        Ok(serde_yaml::to_writer(File::create(path)?, self)?)
    }
}

/// Load the default config, or exit.
pub fn load_or_default_config(path: impl AsRef<Path>) -> Result<Config> {
    let ret = File::open(&path);
    match ret {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Config file not found at path {:?}; writing defaults and exiting.", path.as_ref());
            Config::default().save(path)?;
            std::process::exit(-1);
        },
        Ok(file) => {
            Ok(serde_yaml::from_reader(file)?)
        }
        Err(err) => Err(err)?,
    }
}

impl Default for SimulationSize {
    fn default() -> Self {
        Self {
            width: 500,
            height: 500,
            droplets: 32 * 1000,
        }
    }
}

impl Default for InitSettings {
    fn default() -> Self {
         Self {
            seed: 2.0,
            noise_res: 6,
            noise_amplitude: 1.,
            hill_peak: 1.,
            hill_falloff: 3.,
            n_hills: 4,
        }
    }
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            inertia: 0.07,
            min_slope: 0.01,
            capacity_const: 1.0,
            deposition: 1.0,
            erosion: 0.0000000,
            gravity: 0.1,
            evaporation: 0.05,
        }
    }
}
