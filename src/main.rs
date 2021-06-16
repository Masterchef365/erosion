use anyhow::{Context, Result};
use defaults::FRAMES_IN_FLIGHT;
use std::fs;
use std::path::Path;
use watertender::prelude::*;
mod simulation;
use simulation::*;

struct App {
    rainbow_cube: ManagedMesh,
    pipeline: vk::Pipeline,
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
        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Image uploads
        let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let command_buffer = starter_kit.current_command_buffer(); // TODO: This probably breaks stuff lmaoo

        let (data, info) = read_image("heightmap.png").context("Failed to read image")?;
        let (cube_tex, subresource_range) = starter_kit.staging_buffer.upload_image(
            command_buffer,
            info.width,
            info.height,
            bytemuck::cast_slice(data.as_slice()),
            TEXTURE_FORMAT,
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        // Create image view
        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(cube_tex.instance())
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
        const IMAGE_BINDING: u32 = 1;
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(FRAME_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
                ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
        ];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets((FRAMES_IN_FLIGHT * 2) as _);

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
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<[f32; 4 * 4]>() as u32)];

        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        // Pipeline
        let pipeline = shader(
            core,
            &fs::read("shaders/heightmap.vert.spv")?,
            &fs::read("shaders/unlit_tex.frag.spv")?,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )
        .context("Failed to compile shader")?;

        // Mesh uploads
        assert_eq!(info.width, info.height);
        let size = info.width as i32;
        let scale = 0.1;
        let vertices = dense_grid_verts(size, scale);
        let indices = dense_grid_tri_indices(size);
        let rainbow_cube = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        let size = SimulationSize {
            width: 400,
            height: 400,
            droplets: 10 * 32,
        };

        let init_settings = InitSettings {
            seed: 1.0,
            noise_res: 6,
            noise_amplitude: 1.,
            hill_peak: 1.,
            hill_falloff: 3.,
            n_hills: 4,
        };

        let simulation = ErosionSim::new(
            starter_kit.core.clone(),
            starter_kit.current_command_buffer(),
            &size,
            &init_settings,
        )?;

        Ok(Self {
            camera,
            simulation,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            anim: 0.0,
            pipeline_layout,
            scene_ubo,
            rainbow_cube,
            pipeline,
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

        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.starter_kit.frame]],
                &[],
            );

            // Draw cmds
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            draw_meshes(
                core,
                command_buffer,
                std::slice::from_ref(&&self.rainbow_cube),
            );
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
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl SyncMainLoop for App {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}

fn read_image(path: impl AsRef<Path>) -> Result<(Vec<f32>, png::OutputInfo)> {
    let img = png::Decoder::new(std::fs::File::open(path)?);
    let (info, mut reader) = img.read_info()?;

    const CHANNELS: u32 = 1;
    assert!(info.color_type == png::ColorType::Grayscale);
    assert!(info.bit_depth == png::BitDepth::Eight);

    let mut img_buffer = vec![0; info.buffer_size()];

    assert_eq!(
        info.buffer_size(),
        (info.width * info.height * CHANNELS) as _
    );
    reader.next_frame(&mut img_buffer)?;

    let img_buffer: Vec<f32> = img_buffer
        .into_iter()
        .map(|i| i as f32 / u8::MAX as f32)
        .collect();

    Ok((img_buffer, info))
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
