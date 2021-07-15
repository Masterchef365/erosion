use anyhow::{Context, Result};
use defaults::FRAMES_IN_FLIGHT;
use std::fs;
use std::path::Path;
use watertender::prelude::*;
mod simulation;
use simulation::*;
mod config;
use config::{load_or_default_config, Config};

struct App {
    terrain_mesh: ManagedMesh,
    config: Config,

    terrain_shader: vk::Pipeline,
    droplet_shader: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,

    simulation: ErosionSim,

    scene_ubo: FrameDataUbo<SceneData>,
    camera: MultiPlatformCamera,
    anim: f32,
    starter_kit: StarterKit,
}

fn main() -> Result<()> {
    let info = AppInfo::default().validation(true);
    let vr = std::env::args().count() > 1;
    launch::<App>(info, vr)
}

const TEXTURE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SceneData {
    cameras: [f32; 4 * 4 * 2],
    anim: f32,
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

impl MainLoop for App {
    fn new(core: &SharedCore, mut platform: Platform<'_>) -> Result<Self> {
        let config = load_or_default_config("config.yml").context("Error loading config")?;

        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        let simulation = ErosionSim::new(
            starter_kit.core.clone(),
            starter_kit.current_command_buffer(),
            config.size,
            &config.init,
        )?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Image uploads
        let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // Create image view
        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(simulation.heightmap_image_vk())
            .view_type(vk::ImageViewType::_2D)
            .format(TEXTURE_FORMAT)
            .subresource_range(*subresource_range)
            .build();

        let image_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        // Create sampler
        let create_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(16.)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(0.)
            .build();

        let sampler = unsafe { core.device.create_sampler(&create_info, None, None) }.result()?;

        // Scene data
        let scene_ubo = FrameDataUbo::new(core.clone(), FRAMES_IN_FLIGHT)?;

        // Create descriptor set layout
        const FRAME_DATA_BINDING: u32 = 0;
        const DROPLET_BINDING: u32 = 1;
        const IMAGE_BINDING: u32 = 2;
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(FRAME_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(DROPLET_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
        ];

        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None, None)
        }
        .result()?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
        ];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(FRAMES_IN_FLIGHT as _);

        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Create descriptor sets
        let layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets =
            unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?;

        // Image info
        let image_infos = [vk::DescriptorImageInfoBuilder::new()
            .image_layout(image_layout)
            .image_view(image_view)
            .sampler(sampler)];

        // Droplet buffer info
        let droplet_buffer_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(simulation.droplet_buffer_vk())
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        // Write descriptor sets
        for (frame, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let frame_data_bi = [scene_ubo.descriptor_buffer_info(frame)];
            let writes = [
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(&frame_data_bi)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(FRAME_DATA_BINDING)
                    .dst_array_element(0),
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(&droplet_buffer_info)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(DROPLET_BINDING)
                    .dst_array_element(0),
                vk::WriteDescriptorSetBuilder::new()
                    .image_info(&image_infos)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .dst_binding(IMAGE_BINDING)
                    .dst_array_element(0),
            ];

            unsafe {
                core.device.update_descriptor_sets(&writes, &[]);
            }
        }

        let descriptor_set_layouts = [descriptor_set_layout];

        // Pipeline layout
        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&[])
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        // Pipelines
        let terrain_shader = shader(
            core,
            &fs::read("shaders/heightmap.vert.spv")?,
            &fs::read("shaders/unlit_tex.frag.spv")?,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )
        .context("Failed to compile shader")?;

        let droplet_shader = vert_pulling_shader(
            core,
            &fs::read("shaders/droplet.vert.spv")?,
            &fs::read("shaders/unlit.frag.spv")?,
            vk::PrimitiveTopology::POINT_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )
        .context("Failed to compile shader")?;

        // Mesh uploads
        assert_eq!(config.size.width, config.size.height);
        let size = config.size.width as i32;
        let scale = 1. / (size * 2 + 1) as f32;
        let vertices = dense_grid_verts(size, scale);
        let indices = dense_grid_tri_indices(size);
        let terrain_mesh = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        Ok(Self {
            config,
            camera,
            simulation,
            droplet_shader,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            anim: 0.0,
            pipeline_layout,
            scene_ubo,
            terrain_mesh,
            terrain_shader,
            starter_kit,
        })
    }

    fn frame(
        &mut self,
        frame: Frame,
        core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;

        self.simulation.step(
            command_buffer,
            &self.config.step,
            self.config.control.steps_per_frame,
        )?;

        self.starter_kit
            .begin_render_pass(cmd.command_buffer, frame, [0., 0., 0., 1.]);

        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.starter_kit.frame]],
                &[],
            );

            // Draw terrain
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.terrain_shader,
            );

            draw_meshes(
                core,
                command_buffer,
                std::slice::from_ref(&&self.terrain_mesh),
            );

            // Draw droplets
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.droplet_shader,
            );

            core.device
                .cmd_draw(command_buffer, self.simulation.size().droplets, 1, 0, 0);
        }

        let (ret, cameras) = self.camera.get_matrices(platform)?;

        self.scene_ubo.upload(
            self.starter_kit.frame,
            &SceneData {
                cameras,
                anim: self.anim,
            },
        )?;

        // End draw cmds
        self.starter_kit.end_command_buffer(cmd)?;

        Ok(ret)
    }

    fn swapchain_resize(&mut self, images: Vec<vk::Image>, extent: vk::Extent2D) -> Result<()> {
        self.starter_kit.swapchain_resize(images, extent)
    }

    fn event(
        &mut self,
        mut event: PlatformEvent<'_, '_>,
        _core: &Core,
        mut platform: Platform<'_>,
    ) -> Result<()> {
        use watertender::winit::event::{Event, VirtualKeyCode, WindowEvent};
        if let PlatformEvent::Winit(event) = event {
            if let Event::WindowEvent { event, .. } = event {
                if let WindowEvent::KeyboardInput { input, .. } = event {
                    if let Some(key) = input.virtual_keycode {
                        match key {
                            VirtualKeyCode::S => {
                                self.save_png("out.png").context("Failed to save PNG")?
                            }
                            _ => (),
                        }
                    }
                }
            }
        }

        // TODO: Save image on exit! Also that path should be in the config file.
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.save_png("out.png").expect("Failed to save PNG");
    }
}

impl App {
    fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let img_data: Vec<f32> = self
            .simulation
            .download_heightmap_data(self.starter_kit.command_buffers[0])
            .context("Couldn't download heightmap data")?;
        let img_data = scale_image_data_u8(img_data, 2.);
        let sim_size = self.simulation.size();
        write_gray_png(&img_data, path, sim_size.width, sim_size.height)
    }
}

/// Write a grayscale PNG with this image data
fn write_gray_png(data: &[u8], path: impl AsRef<Path>, width: u32, height: u32) -> Result<()> {
    assert_eq!(data.len() as u32, width * height);

    let file = std::fs::File::create(path)?;
    let ref mut w = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(&data)?;

    Ok(())
}

fn scale_image_data_u8(img_data: Vec<f32>, max_stddevs: f32) -> Vec<u8> {
    use std::cmp::Ordering;
    fn float_cmp(a: &&f32, b: &&f32) -> Ordering {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }

    // Create an iterator of filtered data (no NaNs!)
    let filtered = img_data.iter().filter(|f| f.is_finite());

    // Do some stats to find the standard deviation
    let mean = filtered.clone().sum::<f32>() / img_data.len() as f32;
    let variance =
        filtered.clone().map(|px| (mean - px).powf(2.)).sum::<f32>() / img_data.len() as f32;
    let stddev = variance.sqrt();

    // Subtract pixels from the mean and divide by standard deviation. Then clamp between 0 and 255
    let width = stddev * max_stddevs;
    let px_to_u8 = |px: f32| (((px - mean) / width) * 128. + 128.).max(0.).min(256.) as u8;

    // Export!
    img_data.into_iter().map(px_to_u8).collect()
}

impl SyncMainLoop for App {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}

fn dense_grid_verts(size: i32, scale: f32) -> Vec<Vertex> {
    (-size..=size)
        .map(|x| (-size..=size).map(move |y| (x, y)))
        .flatten()
        .map(|(x, y)| {
            let (x, y) = (x as f32, y as f32);
            let size = size as f32;
            Vertex {
                pos: [x * scale, 0., y * scale],
                color: [((x / size) + 1.) / 2., ((y / size) + 1.) / 2., 0.],
            }
        })
        .collect()
}

fn dense_grid_edge_indices(width: u32) -> impl Iterator<Item = u32> {
    (0..width - 1)
        .map(move |x| (0..width - 1).map(move |y| (x, y)))
        .flatten()
        .map(move |(x, y)| x + y * width)
}

/*
fn dense_grid_wire_indices(size: i32) -> Vec<u32> {
    let width = (size * 2 + 1) as u32;
    let mut indices = Vec::new();
    for base in dense_grid_edge_indices(width) {
        indices.push(base);
        indices.push(base + 1);
        indices.push(base);
        indices.push(base + width);
    }
    // Outer edge
    let outer = width-1;
    for i in 0..outer {
        let edge = (width - 1) * width;
        indices.push(i+edge);
        indices.push(i+edge+1);

        indices.push(i * width + outer);
        indices.push((i + 1) * width + outer);
    }
    indices
}
*/

fn dense_grid_tri_indices(size: i32) -> Vec<u32> {
    let width = (size * 2 + 1) as u32;
    let mut indices = Vec::new();
    for base in dense_grid_edge_indices(width) {
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + width);
        indices.push(base + 1);
        indices.push(base + width + 1);
        indices.push(base + width);
    }
    indices
}

// Build a graphics pipeline compatible with `Vertex` which renders the given primitive
pub fn vert_pulling_shader(
    prelude: &Core,
    vertex_src: &[u8],
    fragment_src: &[u8],
    primitive: vk::PrimitiveTopology,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    // Create shader modules
    let vert_decoded = erupt::utils::decode_spv(vertex_src)?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
    let vertex = unsafe {
        prelude
            .device
            .create_shader_module(&create_info, None, None)
    }
    .result()?;

    let frag_decoded = erupt::utils::decode_spv(fragment_src)?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
    let fragment = unsafe {
        prelude
            .device
            .create_shader_module(&create_info, None, None)
    }
    .result()?;

    // Build pipeline
    let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(primitive)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
        .viewport_count(1)
        .scissor_count(1);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dynamic_states);

    let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_clamp_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)];
    let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let entry_point = std::ffi::CString::new("main")?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vertex)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(fragment)
            .name(&entry_point),
    ];

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let create_info = vk::GraphicsPipelineCreateInfoBuilder::new()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        prelude
            .device
            .create_graphics_pipelines(None, &[create_info], None)
    }
    .result()?[0];

    unsafe {
        prelude.device.destroy_shader_module(Some(fragment), None);
        prelude.device.destroy_shader_module(Some(vertex), None);
    }

    Ok(pipeline)
}
