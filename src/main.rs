use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk::{
    self, ComponentMapping, ImageViewCreateFlags, PipelineInputAssemblyStateCreateFlags,
    PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateFlags,
    PipelineVertexInputStateCreateInfo, SampleCountFlags, ShaderModuleCreateFlags,
    SwapchainCreateInfoKHR,
};
use platform::create_surface;
use std::collections::{BTreeSet, HashSet};
use std::ffi::{CStr, CString};
use std::ptr;
use validation::{ValidationsRequested, VALIDATION_ON};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
mod platform;
mod validation;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const WINDOW_NAME: &str = "Vulkan test";
const ENGINE_NAME: &str = "No engine";

const REQUIRED_VALIDATIONS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
fn device_required_features() -> Vec<&'static CStr> {
    vec![ash::extensions::khr::Swapchain::name()]
}

struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct DeviceWithCapabilities {
    device: vk::PhysicalDevice,
    swapchain: SwapchainSupport,
    queue_family: QueueFamilyIndices,
}

struct QueueFamilyIndicesIncomplete {
    graphics_family: Option<usize>,
    present_family: Option<usize>,
}

impl QueueFamilyIndicesIncomplete {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }

    fn get_complete(self) -> Option<QueueFamilyIndices> {
        if let (Some(graphics), Some(present)) = (self.graphics_family, self.present_family) {
            Some(QueueFamilyIndices {
                graphics_family: graphics as u32,
                present_family: present as u32,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl QueueFamilyIndices {
    pub fn as_array(&self) -> [u32; 2] {
        [self.graphics_family, self.present_family]
    }
}

struct Surface {
    surface: vk::SurfaceKHR,
    loader: ash::extensions::khr::Surface,
}

impl Surface {
    fn new(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> Surface {
        let surface =
            unsafe { create_surface(entry, instance, window) }.expect("Failed to create Surface");
        let loader = ash::extensions::khr::Surface::new(entry, instance);
        Surface { surface, loader }
    }

    fn khr(&self) -> vk::SurfaceKHR {
        self.surface
    }

    unsafe fn destroy(&self) {
        self.loader.destroy_surface(self.surface, None);
    }
}

struct Swapchain {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    image_format: vk::Format,
    extent: vk::Extent2D,
}

impl Swapchain {
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        create_info: &SwapchainCreateInfoKHR,
    ) -> Swapchain {
        let loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe { loader.create_swapchain(create_info, None) }
            .expect("Failed to create swapchain");
        Swapchain {
            swapchain,
            loader,
            image_format: create_info.image_format,
            extent: create_info.image_extent,
        }
    }

    unsafe fn destroy(&self) {
        self.loader.destroy_swapchain(self.swapchain, None);
    }
}

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: Surface,
    swapchain: Swapchain,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.expect("Could not load Vulkan library");
        let validations = ValidationsRequested::new(&REQUIRED_VALIDATIONS[..]);
        let instance = create_instance(&entry, &validations);
        let surface = Surface::new(&entry, &instance, window);
        let physical_device = pick_physical_device(&instance, &surface);
        let (device, queue_indices) =
            create_logical_device(&instance, &physical_device, &validations, &surface);
        let graphics_queue = unsafe { device.get_device_queue(queue_indices.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_indices.present_family, 0) };
        let swapchain = create_swapchain(&instance, &device, &physical_device, &surface);
        let images = unsafe { swapchain.loader.get_swapchain_images(swapchain.swapchain) }
            .expect("Failed to get images");
        let image_views = create_image_views(&device, &swapchain, &images);
        let render_pass = create_render_pass(&device, &swapchain);
        VulkanApp {
            entry,
            instance,
            device,
            graphics_queue,
            present_queue,
            surface,
            swapchain,
            image_views,
            render_pass,
        }
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title(WINDOW_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window")
    }

    pub fn main_loop(self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            dbg!();
                            *control_flow = ControlFlow::Exit
                        }
                        _ => {}
                    },
                },
                _ => {}
            },
            _ => (),
        })
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None);
            self.image_views
                .iter()
                .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.swapchain.destroy();
            self.surface.destroy();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        };
    }
}

fn create_instance(entry: &ash::Entry, validations: &ValidationsRequested) -> ash::Instance {
    if VALIDATION_ON && !validations.check_support(entry) {
        panic!("Some validation layers requested are not available");
    }
    let window_name = CString::new(WINDOW_NAME).unwrap();
    let engine_name = CString::new(ENGINE_NAME).unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: window_name.as_ptr(),
        application_version: vk::make_version(0, 1, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_version(0, 1, 0),
        api_version: vk::API_VERSION_1_2,
    };
    let required_extensions = platform::required_extension_names();

    let creation_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: if VALIDATION_ON {
            validations.layers_ptr().len() as u32
        } else {
            0
        },
        pp_enabled_layer_names: if VALIDATION_ON {
            validations.layers_ptr().as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: required_extensions.len() as u32,
        pp_enabled_extension_names: required_extensions.as_ptr(),
    };
    unsafe {
        entry
            .create_instance(&creation_info, None)
            .expect("Failed to create instance")
    }
}

fn pick_physical_device(instance: &ash::Instance, surface: &Surface) -> DeviceWithCapabilities {
    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
    log::info!("Found {} GPUs in the system", physical_devices.len());
    physical_devices
        .into_iter()
        .map(|device| rate_physical_device_suitability(instance, device, surface))
        .flatten()
        .max_by_key(|(score, _)| *score)
        .expect("No compatible physical devices found")
        .1
}

fn rate_physical_device_suitability(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> Option<(u32, DeviceWithCapabilities)> {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_features = unsafe { instance.get_physical_device_features(physical_device) };
    let score = match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
        vk::PhysicalDeviceType::CPU => 1,
        vk::PhysicalDeviceType::OTHER => 10,
        _ => 10,
    };
    if device_support_requested_extensions(instance, physical_device) {
        let queue_family = find_queue_families(instance, physical_device, surface);
        let swapchain = query_swapchain_capabilities(surface, physical_device);
        if let Some(queue_family) = queue_family.get_complete() {
            if !swapchain.formats.is_empty() && !swapchain.present_modes.is_empty() {
                Some((
                    score,
                    DeviceWithCapabilities {
                        device: physical_device,
                        swapchain,
                        queue_family,
                    },
                ))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

fn device_support_requested_extensions(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
) -> bool {
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(device) }
        .expect("Failed to get device extensions")
        .into_iter()
        .map(|x| validation::cchars_to_string(&x.extension_name))
        .collect::<HashSet<_>>();
    !device_required_features()
        .into_iter()
        .map(|x| x.to_str().unwrap().to_string())
        .any(|x| !available_extensions.contains(&x))
}

fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> QueueFamilyIndicesIncomplete {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let graphics_family = queue_families
        .iter()
        .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    let present_family = (0..queue_families.len()).into_iter().find(|x| unsafe {
        surface
            .loader
            .get_physical_device_surface_support(physical_device, *x as u32, surface.khr())
            .unwrap()
    });
    QueueFamilyIndicesIncomplete {
        graphics_family,
        present_family,
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    device: &DeviceWithCapabilities,
    validations: &ValidationsRequested,
    surface: &Surface,
) -> (ash::Device, QueueFamilyIndices) {
    let physical_device = device.device;
    let queue_families = device.queue_family;
    let queue_families_set = queue_families
        .as_array()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut queue_create_infos = Vec::with_capacity(queue_families_set.len());
    let queue_priority = 1.0;
    for queue_index in queue_families_set {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_index,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
        };
        queue_create_infos.push(queue_create_info);
    }

    let physical_features = vk::PhysicalDeviceFeatures {
        ..Default::default()
    };
    let required_device_extensions = device_required_features()
        .into_iter()
        .map(|x| x.as_ptr())
        .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: if VALIDATION_ON {
            validations.layers_ptr().len() as u32
        } else {
            0
        },
        pp_enabled_layer_names: if VALIDATION_ON {
            validations.layers_ptr().as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: required_device_extensions.len() as u32,
        pp_enabled_extension_names: required_device_extensions.as_ptr(),
        p_enabled_features: &physical_features,
    };
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("Failed to create logical device");
    (device, queue_families)
}

fn query_swapchain_capabilities(
    surface: &Surface,
    physical_device: vk::PhysicalDevice,
) -> SwapchainSupport {
    let (surface, loader) = (surface.surface, &surface.loader);
    let capabilities =
        unsafe { loader.get_physical_device_surface_capabilities(physical_device, surface) }
            .expect("could not get surface capabilities");
    let formats = unsafe { loader.get_physical_device_surface_formats(physical_device, surface) }
        .expect("Could not get surface formats");
    let present_modes =
        unsafe { loader.get_physical_device_surface_present_modes(physical_device, surface) }
            .expect("Failed to get present modes");
    SwapchainSupport {
        capabilities,
        formats,
        present_modes,
    }
}

fn create_swapchain(
    instance: &ash::Instance,
    ldevice: &ash::Device,
    device: &DeviceWithCapabilities,
    surface: &Surface,
) -> Swapchain {
    let capabilities = &device.swapchain.capabilities;
    let format = *device
        .swapchain
        .formats
        .iter()
        .find(|sf| {
            sf.format == vk::Format::B8G8R8A8_SRGB
                && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(device.swapchain.formats.first().unwrap());
    let present_mode = *device
        .swapchain
        .present_modes
        .iter()
        .find(|&x| *x == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(&vk::PresentModeKHR::FIFO);
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: WINDOW_WIDTH.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: WINDOW_HEIGHT.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };
    let image_count = if capabilities.max_image_count == 0 {
        capabilities.min_image_count + 1
    } else {
        capabilities
            .max_image_count
            .max(capabilities.min_image_count + 1)
    };
    let queue_families_indices = [
        device.queue_family.graphics_family,
        device.queue_family.present_family,
    ];
    let (image_sharing_mode, queue_family_index_count, p_queue_family_indices) =
        if device.queue_family.graphics_family != device.queue_family.present_family {
            (
                vk::SharingMode::CONCURRENT,
                2,
                queue_families_indices.as_ptr(),
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
        };
    let create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::default(),
        surface: surface.khr(),
        min_image_count: image_count,
        image_format: format.format,
        image_color_space: format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode,
        queue_family_index_count,
        p_queue_family_indices,
        pre_transform: capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        old_swapchain: vk::SwapchainKHR::null(),
    };
    Swapchain::new(instance, ldevice, &create_info)
}

fn create_image_views(
    device: &ash::Device,
    swapchain: &Swapchain,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    let mut retval = Vec::with_capacity(images.len());
    for image in images {
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: ImageViewCreateFlags::default(),
            image: *image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain.image_format,
            components: ComponentMapping::default(),
            subresource_range,
        };
        retval.push(
            unsafe { device.create_image_view(&create_info, None) }
                .expect("Failed to create Image View"),
        );
    }
    retval
}

fn create_render_pass(device: &ash::Device, swapchain: &Swapchain) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription {
        flags: Default::default(),
        format: swapchain.image_format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };
    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription {
        flags: Default::default(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_ref,
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    };
    let render_pass_ci = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        attachment_count: 1,
        p_attachments: &color_attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 0,
        p_dependencies: ptr::null(),
    };
    unsafe { device.create_render_pass(&render_pass_ci, None) }
        .expect("Failed to create render pass")
}

fn create_graphics_pipeline(device: &ash::Device, swapchain: &Swapchain) {
    let main_str = CString::new("main").unwrap();
    let vertex_shader = include_bytes!("../target/shaders/triangle.vert.spv");
    let fragment_shader = include_bytes!("../target/shaders/triangle.frag.spv");
    let vertex_module = create_shader_module(device, vertex_shader);
    let fragment_module = create_shader_module(device, fragment_shader);
    let vertex_stage_ci = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: PipelineShaderStageCreateFlags::default(),
        stage: vk::ShaderStageFlags::VERTEX,
        module: vertex_module,
        p_name: main_str.as_ptr(),
        p_specialization_info: ptr::null(),
    };
    let fragment_stage_ci = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: PipelineShaderStageCreateFlags::default(),
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: fragment_module,
        p_name: main_str.as_ptr(),
        p_specialization_info: ptr::null(),
    };
    let vertex_info_ci = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        vertex_binding_description_count: 0,
        p_vertex_binding_descriptions: ptr::null(),
        vertex_attribute_description_count: 0,
        p_vertex_attribute_descriptions: ptr::null(),
    };
    let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
    };
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain.extent.width as f32,
        height: swapchain.extent.height as f32,
        min_depth: 0.0,
        max_depth: 0.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain.extent,
    };
    let viewport_state_ci = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,
    };
    let rasterizer_ci = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::TRUE,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::CLOCKWISE,
        depth_bias_enable: vk::FALSE,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0,
        line_width: 1.0,
    };
    let multisampling_ci = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 1.0,
        p_sample_mask: ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
    };
    let blending_settings = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::all(),
    };
    let blending_ci = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: 1,
        p_attachments: &blending_settings,
        blend_constants: [0.0; 4],
    };
    let dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
    let dynamic_state_ci = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        dynamic_state_count: 2,
        p_dynamic_states: dynamic_states.as_ptr(),
    };
    let layout_ci = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        set_layout_count: 0,
        p_set_layouts: ptr::null(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };
    let layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }
        .expect("Failed to create layout");
    unsafe {
        device.destroy_pipeline_layout(layout, None);
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
}

fn create_shader_module(device: &ash::Device, shader_code: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: ShaderModuleCreateFlags::default(),
        code_size: shader_code.len(),
        p_code: shader_code.as_ptr() as *const u32,
    };
    unsafe { device.create_shader_module(&create_info, None) }
        .expect("Failed to create shader module")
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);
    let app = VulkanApp::new(&window);
    app.main_loop(event_loop);
}
