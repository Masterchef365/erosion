use anyhow::{Result, Context};
use watertender::{
    memory::{ManagedBuffer, ManagedImage, UsageFlags},
    vk, SharedCore, Core,
};

pub const HEIGHT_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub const EROSION_MAP_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

const SETTINGS_BINDING: u32 = 0;
const DROPLETS_BINDING: u32 = 1;
const HEIGHT_MAP_BINDING: u32 = 2;
const EROSION_MAP_BINDING: u32 = 3;
//const HEIGHT_MAP_SAMPLER_BINDING: u32 = 4;

const KERNEL_LOCAL_X: u32 = 32;
const KERNEL_LOCAL_Y: u32 = 32;

pub struct ErosionSim {
    droplets: ManagedBuffer,
    erosion: ManagedImage,
    heightmap: ManagedImage,
    settings_buf: ManagedBuffer,

    erosion_view: vk::ImageView,
    heightmap_view: vk::ImageView,

    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,

    init_droplets: vk::Pipeline,
    init_heightmap: vk::Pipeline,
    sim_step: vk::Pipeline,
    erosion_blur: vk::Pipeline,

    sim_size: SimulationSize,

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

unsafe impl bytemuck::Zeroable for InitSettings {}
unsafe impl bytemuck::Pod for InitSettings {}

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

unsafe impl bytemuck::Zeroable for SimulationSettings {}
unsafe impl bytemuck::Pod for SimulationSettings {}

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

#[derive(Copy, Clone, Debug)]
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
        sim_size: SimulationSize,
        init_settings: &InitSettings,
    ) -> Result<Self> {
        // Create erosion and heightmap images
        let extent = vk::Extent3DBuilder::new()
            .width(sim_size.width)
            .height(sim_size.height)
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
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlagBits::_1);

        let erosion = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;
        let heightmap = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;

        // Create droplet buffer
        let ci = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(sim_size.droplets as u64 * std::mem::size_of::<Droplet>() as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);

        let droplets = ManagedBuffer::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;

        // Create settings buffer
        let init_settings_size = std::mem::size_of::<InitSettings>() as u64;
        let sim_settings_size = std::mem::size_of::<SimulationSettings>() as u64;

        let ci = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(init_settings_size.max(sim_settings_size))
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);

        let settings_buf = ManagedBuffer::new(core.clone(), ci, UsageFlags::UPLOAD)?;

        // Descriptors
        // Pool:
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1),
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1),
            vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2),
            /*vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1),*/
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
                .binding(HEIGHT_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(EROSION_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            /*vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(HEIGHT_MAP_SAMPLER_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),*/
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

        // Pipeline
        let create_info =
            vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&[])
            .set_layouts(&descriptor_set_layouts);
        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        let init_droplets = load_pipeline(&core, "kernels/init_droplets.comp.spv", pipeline_layout)?;
        let init_heightmap = load_pipeline(&core, "kernels/init_heightmap.comp.spv", pipeline_layout)?;
        let sim_step = load_pipeline(&core, "kernels/sim_step.comp.spv", pipeline_layout)?;
        let erosion_blur = load_pipeline(&core, "kernels/erosion_blur.comp.spv", pipeline_layout)?;

        // Create image views
        let img_subresource = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let cm = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };

        // Create heightmap view
        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(heightmap.instance())
            .view_type(vk::ImageViewType::_2D)
            .format(HEIGHT_MAP_FORMAT)
            .components(cm)
            .subresource_range(img_subresource);
        let heightmap_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        let heightmap_image_infos = [vk::DescriptorImageInfoBuilder::new()
            .image_view(heightmap_view)
            .image_layout(vk::ImageLayout::GENERAL)];

        // Create erosion view
        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(erosion.instance())
            .view_type(vk::ImageViewType::_2D)
            .format(HEIGHT_MAP_FORMAT)
            .components(cm)
            .subresource_range(img_subresource);
        let erosion_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        let erosion_image_infos = [vk::DescriptorImageInfoBuilder::new()
            .image_view(erosion_view)
            .image_layout(vk::ImageLayout::GENERAL)];

        // Descriptor buffer vies
        let droplets_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(droplets.instance())
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let settings_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(settings_buf.instance())
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        // Write descriptor sets
        let desc_set_writes = [
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(SETTINGS_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&settings_info),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(DROPLETS_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&droplets_info),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(HEIGHT_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&heightmap_image_infos),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(EROSION_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&erosion_image_infos),
        ];

        unsafe {
            core.device.update_descriptor_sets(&desc_set_writes, &[]);
        }

        let mut instance = Self {
            sim_size,
            erosion_view,
            heightmap_view,
            settings_buf,
            descriptor_set,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            erosion,
            heightmap,
            droplets,
            core,
            init_heightmap,
            init_droplets,
            sim_step,
            erosion_blur,
        };

        instance.reset(cmd, init_settings)?;
        Ok(instance)
    }

    pub fn step(
        &mut self,
        cmd: vk::CommandBuffer,
        settings: &SimulationSettings,
        iters: u32,
    ) -> Result<()> {
        self.settings_buf.write_bytes(0, bytemuck::cast_slice(std::slice::from_ref(settings)))?;

        let img_subresource = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        unsafe {
            // Transition heightmap and erosion images to GENERAL
            let image_memory_barriers = [
                vk::ImageMemoryBarrierBuilder::new()
                    .image(self.heightmap.instance())
                    .subresource_range(img_subresource)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL),
                vk::ImageMemoryBarrierBuilder::new()
                    .image(self.erosion.instance())
                    .subresource_range(img_subresource)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
            ];

            self.core.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                None,
                &[],
                &[],
                &image_memory_barriers,
            );

            // Bind descriptor set
            self.core.device.cmd_bind_descriptor_sets(
                cmd, 
                vk::PipelineBindPoint::COMPUTE, 
                self.pipeline_layout, 
                0, 
                &[self.descriptor_set], 
                &[]
            );

            // Iteration loop
            for _ in 0..iters {
                // Clear erosion image
                self.core.device.cmd_clear_color_image(
                    cmd,
                    self.erosion.instance(),
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.; 4],
                    },
                    &[img_subresource.into_builder()],
                );

                // Launch sim step kernel
                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.sim_step,
                );
                self.core.device.cmd_dispatch(
                    cmd,
                    (self.sim_size.droplets / KERNEL_LOCAL_X) + 1,
                    1,
                    1,
                );

                // Launch erosion kernel
                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.erosion_blur,
                );
                self.core.device.cmd_dispatch(
                    cmd,
                    (self.sim_size.width / KERNEL_LOCAL_X) + 1,
                    (self.sim_size.height / KERNEL_LOCAL_X) + 1,
                    1,
                );
                
                // TODO: Make memory available
                self.core.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    None,
                    &[],
                    &[],
                    &[],
                );
            }

            // Transition heightmap and erosion images to SHADER_READ_ONLY_OPTIMAL
            let image_memory_barriers = [
                vk::ImageMemoryBarrierBuilder::new()
                    .image(self.heightmap.instance())
                    .subresource_range(img_subresource)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                vk::ImageMemoryBarrierBuilder::new()
                    .image(self.erosion.instance())
                    .subresource_range(img_subresource)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            ];

            self.core.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                None,
                &[],
                &[],
                &image_memory_barriers,
            );

        }

        Ok(())
    }

    pub fn reset(&mut self, cmd: vk::CommandBuffer, settings: &InitSettings) -> Result<()> {
        // Update settings buffer (TODO: Barrier?)
        self.settings_buf.write_bytes(0, bytemuck::cast_slice(std::slice::from_ref(settings)))?;

        let img_subresource = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let image_memory_barriers = [
            vk::ImageMemoryBarrierBuilder::new()
                .image(self.heightmap.instance())
                .subresource_range(img_subresource)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL),
            vk::ImageMemoryBarrierBuilder::new()
                .image(self.erosion.instance())
                .subresource_range(img_subresource)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
        ];

        unsafe {
            self.core.device.reset_command_buffer(cmd, None).result()?;
            let bi = vk::CommandBufferBeginInfoBuilder::new();
            self.core.device.begin_command_buffer(cmd, &bi).result()?;

            self.core.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                None,
                &[],
                &[],
                &image_memory_barriers,
            );

            self.core.device.cmd_bind_descriptor_sets(
                cmd, 
                vk::PipelineBindPoint::COMPUTE, 
                self.pipeline_layout, 
                0, 
                &[self.descriptor_set], 
                &[]
            );

            self.core.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.init_droplets);
            self.core.device.cmd_dispatch(cmd, (self.sim_size.droplets / KERNEL_LOCAL_X) + 1, 1, 1);

            self.core.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.init_heightmap);
            self.core.device.cmd_dispatch(cmd, (self.sim_size.width / KERNEL_LOCAL_X) + 1, (self.sim_size.height / KERNEL_LOCAL_Y) + 1, 1);

            self.core.device.end_command_buffer(cmd).result()?;

            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfoBuilder::new()
                .command_buffers(&command_buffers);

            self.core
                .device
                .queue_submit(self.core.queue, &[submit_info], None)
                .result()?;

            self.core.device.queue_wait_idle(self.core.queue).result()?;
        }

        Ok(())
    }

    pub fn droplet_buffer_vk(&self) -> vk::Buffer {
        self.droplets.instance()
    }

    pub fn heightmap_image_vk(&self) -> vk::Image {
        self.heightmap.instance()
    }

    pub fn size(&self) -> SimulationSize {
        self.sim_size
    }

    //pub fn download_droplets(&mut self) -> Vec<Particle> {}
    //pub fn download_heightmap(&mut self) -> Vec<f32> {}

    //pub fn upload_heightmap(&mut self, data: &[f32]);
    //pub fn upload_droplets(&mut self, droplets: &[Particle]);
}

fn load_pipeline(core: &Core, path: &str, pipeline_layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
    let device = &core.device;

    // Load shader
    let shader_spirv = std::fs::read(path).context("Failed to read")?;
    let shader_decoded = erupt::utils::decode_spv(&shader_spirv).context("Decode failed")?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
    let shader_module =
        unsafe { device.create_shader_module(&create_info, None, None) }.result()?;

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
