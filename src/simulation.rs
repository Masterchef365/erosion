use watertender::{
    memory::{ManagedBuffer, ManagedImage},
    staging_buffer::StagingBuffer,
    SharedCore, vk,
};
use anyhow::Result;

pub struct ErosionSim {
    droplets: ManagedBuffer,
    erosion: ManagedImage,
    heightmap: ManagedImage,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InitSettings {
    /// Random seed
    pub seed: u64,
    /// Noise scale
    pub noise_scale: f32,
    /// Noise vertical amplitude
    pub noise_amplitude: f32,
    /// Number of hills (randomly placed)
    pub n_hills: u32,
    /// Falloff rate for the curve
    pub hill_falloff: f32,
}

// TODO: Builder pattern?
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimulationSettings {
    /// Inertia
    pub inertia: f32,
    /// Minimum slope for capacity calculation
    pub min_slope: f32,
    /// Capacity for droplets to carry material
    pub capacity_const: f32,
    /// Sediment dropped beyond capacity
    pub deposition: f32,
    /// Sediment picked up under capacity
    pub erosion: f32,
    /// Force of gravity
    pub gravity: f32,
    /// Evaporation rate
    pub evaporation: f32, // TODO: 1- evap for speed!
}

/*
pub struct Droplet {
    /// Position
    pos: [f32; 2],
    /// Direction
    dir: [f32; 2],
    /// Velocity
    vel: f32,
    /// Water
    water: f32,
    /// Sediment
    sediment: f32,
}
*/

impl ErosionSim {
    pub fn new(core: SharedCore, cmd: vk::CommandBuffer, width: u32, height: u32, settings: &InitSettings) -> Result<Self> {

        //let instance = Self {
        //}
        let instance: Self = todo!();

        instance.reset(cmd, settings)?;
        Ok(instance)
    }

    pub fn step(&mut self, cmd: vk::CommandBuffer, settings: &SimulationSettings, iters: u32) -> Result<()> {
        todo!()
    }

    pub fn reset(&mut self, cmd: vk::CommandBuffer, settings: &InitSettings) -> Result<()> {
        todo!()
    }

    pub fn droplet_buffer_vk(&mut self) -> vk::Buffer {
        self.droplets.instance()
    }

    pub fn heightmap_image_vk(&mut self) -> vk::Image {
        self.heightmap.instance()
    }

    //pub fn download_particles(&mut self) -> Vec<Particle> {}
    //pub fn download_heightmap(&mut self) -> Vec<f32> {}

    //pub fn upload_heightmap(&mut self, data: &[f32]);
    //pub fn upload_particles(&mut self, particles: &[Particle]);
}
