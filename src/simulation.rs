use anyhow::{Result, Context};
use watertender::{
    memory::{ManagedBuffer, ManagedImage, UsageFlags},
    staging_buffer::StagingBuffer,
    vk, SharedCore, Core,
};

pub const HEIGHT_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub const EROSION_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

const SETTINGS_BINDING: u32 = 0;
const DROPLETS_BINDING: u32 = 1;
const HEIGHT_MAP_SAMPLER_BINDING: u32 = 2;
const HEIGHT_MAP_BINDING: u32 = 3;
const EROSION_MAP_BINDING: u32 = 4;

pub struct ErosionSim {
    droplets: ManagedBuffer,
    erosion: ManagedImage,
    heightmap: ManagedImage,

    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,

    init_droplets: vk::Pipeline,
    init_heightmap: vk::Pipeline,
    sim_step: vk::Pipeline,
    erosion_blur: vk::Pipeline,

    core: SharedCore,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InitSettings {
    /// Random seed
    pub seed: f32,
    /// Noise resolution
    pub noise_res: i32,
    /// Noise vertical amplitude
    pub noise_amplitude: f32,
    /// Hill peak height
    pub hill_peak: f32,
    /// Falloff rate for the curve
    pub hill_falloff: f32,
    /// Number of hills to consider
    pub n_hills: i32,
}

// TODO: Builder pattern?
#[repr(C)]
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

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
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

pub struct SimulationSize {
    pub width: u32,
    pub height: u32,
    //pub hills: u32,
    pub droplets: u32,
}

impl ErosionSim {
    pub fn new(
        core: SharedCore,
        cmd: vk::CommandBuffer,
        size: &SimulationSize,
        settings: &InitSettings,
    ) -> Result<Self> {
        // Create erosion and heightmap images
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

        let erosion = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;
        let heightmap = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;

        // Create droplet buffer
        let ci = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(size.droplets as u64 * std::mem::size_of::<Droplet>() as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);

        let droplets = ManagedBuffer::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;

        // Descriptors
        // Pool:
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2),
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1),
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1),
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
        ];
        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Layout:
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(SETTINGS_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(DROPLETS_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(HEIGHT_MAP_SAMPLER_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(HEIGHT_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(EROSION_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { core.device.create_descriptor_set_layout(&create_info, None, None) }.result()?;

        // Set:
        let descriptor_set_layouts = [descriptor_set_layout];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_set = unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?[0];

        // Pipelines
        let init_droplets = load_pipeline(&core, "kernels/init_droplets.comp.spv", &descriptor_set_layouts)?;
        let init_heightmap = load_pipeline(&core, "kernels/init_heightmap.comp.spv", &descriptor_set_layouts)?;
        let sim_step = load_pipeline(&core, "kernels/sim_step.comp.spv", &descriptor_set_layouts)?;
        let erosion_blur = load_pipeline(&core, "kernels/erosion_blur.comp.spv", &descriptor_set_layouts)?;

        let mut instance = Self {
            descriptor_set,
            descriptor_pool,
            descriptor_set_layout,
            erosion,
            heightmap,
            droplets,
            core,
            init_heightmap,
            init_droplets,
            sim_step,
            erosion_blur,
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

    //pub fn download_droplets(&mut self) -> Vec<Particle> {}
    //pub fn download_heightmap(&mut self) -> Vec<f32> {}

    //pub fn upload_heightmap(&mut self, data: &[f32]);
    //pub fn upload_droplets(&mut self, droplets: &[Particle]);
}

fn load_pipeline(core: &Core, path: &str, descriptor_set_layouts: &[vk::DescriptorSetLayout]) -> Result<vk::Pipeline> {
    let device = &core.device;

    // Load shader
    let shader_spirv = std::fs::read(path).context("Failed to read")?;
    let shader_decoded = erupt::utils::decode_spv(&shader_spirv).context("Decode failed")?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
    let shader_module =
        unsafe { device.create_shader_module(&create_info, None, None) }.result()?;

    // Pipeline
    let push_constant_ranges = [
        vk::PushConstantRangeBuilder::new()
            .offset(0)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(2 * std::mem::size_of::<i32>() as u32)
    ];

    let create_info =
        vk::PipelineLayoutCreateInfoBuilder::new()
        .push_constant_ranges(&push_constant_ranges)
        .set_layouts(&descriptor_set_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&create_info, None, None) }.result()?;

    let entry_point = std::ffi::CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk::ShaderStageFlagBits::COMPUTE)
        .module(shader_module)
        .name(&entry_point)
        .build();
    let create_info = vk::ComputePipelineCreateInfoBuilder::new()
        .stage(stage)
        .layout(pipeline_layout);
    let pipeline =
        unsafe { device.create_compute_pipelines(None, &[create_info], None) }.result()?[0];

    unsafe {
        device.destroy_shader_module(Some(shader_module), None);
    }

    return Ok(pipeline);
}
