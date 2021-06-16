use anyhow::Result;
use watertender::{
    memory::{ManagedBuffer, ManagedImage, UsageFlags},
    staging_buffer::StagingBuffer,
    vk, SharedCore,
};

pub const HEIGHT_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub const EROSION_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

pub struct ErosionSim {
    hills: ManagedBuffer,
    droplets: ManagedBuffer,
    erosion: ManagedImage,
    heightmap: ManagedImage,

    init_particles: vk::Pipeline,
    init_heightmap: vk::Pipeline,
    sim_step: vk::Pipeline,
    erosion_blur: vk::Pipeline,
    core: SharedCore,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InitSettings {
    /// Random seed
    pub seed: u64,
    /// Noise scale
    pub noise_scale: f32,
    /// Noise vertical amplitude
    pub noise_amplitude: f32,
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

struct SimulationSize {
    width: u32,
    height: u32,
    hills: u32,
    droplets: u32,
}

impl ErosionSim {
    pub fn new(
        core: SharedCore,
        cmd: vk::CommandBuffer,
        size: SimulationSize,
        settings: &InitSettings,
    ) -> Result<Self> {
        // Image settings
        let extent = vk::Extent3DBuilder::new()
            .width(size.width)
            .height(size.height)
            .depth(1)
            .build();

        debug_assert_eq!(EROSION_MAP_FORMAT, HEIGHT_MAP_FORMAT, "I'm lazy");
        let ci = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(EROSION_MAP_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlagBits::_1);

        let erosion = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS);
        let heightmap = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS);

        let instance = Self {
            erosion,
            heightmap,
            core,
        };

        instance.reset(cmd, settings)?;
        Ok(instance)
    }

    pub fn step(
        &mut self,
        cmd: vk::CommandBuffer,
        settings: &SimulationSettings,
        iters: u32,
    ) -> Result<()> {
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
