use ash::vk;
use cgmath::{perspective, Deg, Matrix4, Point3, Vector3 as Vec3};
use memoffset::offset_of;
use platform::create_surface;
use std::collections::{BTreeSet, HashSet};
use std::ffi::{CStr, CString};
use std::ptr;
use std::time::Instant;
use validation::{ValidationsRequested, VALIDATION_ON};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
mod platform;
mod validation;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const WINDOW_NAME: &str = "Vulkan test";
const ENGINE_NAME: &str = "No engine";
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const REQUIRED_VALIDATIONS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
fn device_required_features() -> Vec<&'static CStr> {
    vec![ash::extensions::khr::Swapchain::name()]
}

const VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vec3 {
            x: -0.5,
            y: -0.5,
            z: 0.0,
        },
        color: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    },
    Vertex {
        position: Vec3 {
            x: 0.5,
            y: -0.5,
            z: 0.0,
        },
        color: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    },
    Vertex {
        position: Vec3 {
            x: 0.5,
            y: 0.5,
            z: 0.0,
        },
        color: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    },
    Vertex {
        position: Vec3 {
            x: -0.5,
            y: 0.5,
            z: 0.0,
        },
        color: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    },
];

const INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

#[repr(C, packed)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct PDeviceAndQueues {
    device: vk::PhysicalDevice,
    queue_family: QueueFamilyIndices,
}

struct QueueFamilyIndicesIncomplete {
    graphics_family: Option<usize>,
    present_family: Option<usize>,
}

impl QueueFamilyIndicesIncomplete {
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
        create_info: &vk::SwapchainCreateInfoKHR,
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

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct Vertex {
    position: Vec3<f32>,
    color: Vec3<f32>,
}

impl Vertex {
    fn new(values: [f32; 6]) -> Vertex {
        Vertex {
            position: Vec3 {
                x: values[0],
                y: values[1],
                z: values[2],
            },
            color: Vec3 {
                x: values[3],
                y: values[4],
                z: values[5],
            },
        }
    }

    fn binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ]
    }
}

struct DrawData {
    vertex_buffer: vk::Buffer,
    vb_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    ib_memory: vk::DeviceMemory,
    descriptor_sets: Vec<vk::DescriptorSet>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

struct VulkanApp {
    // dropping an Entry causes every call that uses Surface to SEGFAULT on linux
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: PDeviceAndQueues,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    window: Window,
    surface: Surface,
    swapchain: Swapchain,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    sem_img_available: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    sem_render_finish: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    acquire_inflight: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    images_inflight: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    draw_data: DrawData,
    u_buf: Vec<vk::Buffer>,
    u_mem: Vec<vk::DeviceMemory>,
    start_time: Instant,
    inflight_frame_no: usize,
    resized: bool,
}

impl VulkanApp {
    pub fn new(window: Window) -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.expect("Could not load Vulkan library");
        let validations = ValidationsRequested::new(&REQUIRED_VALIDATIONS[..]);
        let instance = create_instance(&entry, &validations);
        let surface = Surface::new(&entry, &instance, &window);
        let physical_device = pick_physical_device(&instance, &surface);
        let swapchain_support = query_swapchain_capabilities(&surface, physical_device.device);
        let (device, queue_indices) =
            create_logical_device(&instance, &physical_device, &validations);
        let graphics_queue = unsafe { device.get_device_queue(queue_indices.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_indices.present_family, 0) };
        let swapchain = create_swapchain(
            &instance,
            &device,
            &swapchain_support,
            &physical_device.queue_family,
            &surface,
        );
        let images = unsafe { swapchain.loader.get_swapchain_images(swapchain.swapchain) }
            .expect("Failed to get images");
        let image_views = create_image_views(&device, &swapchain, &images);
        let render_pass = create_render_pass(&device, &swapchain);
        let descriptor_layout = create_descriptor_set_layout(&device);
        let descriptor_pool = create_descriptor_pool(&device, images.len() as u32);
        let pipeline_layout = create_pipeline_layout(&device, &[descriptor_layout]);
        let pipeline = create_graphics_pipeline(&device, &swapchain, render_pass, pipeline_layout);
        let (u_buf, u_mem) =
            allocate_uniform_buffers(&instance, physical_device.device, &device, images.len());
        let descriptor_sets = create_descriptor_sets(
            &device,
            images.len() as u32,
            descriptor_pool,
            descriptor_layout,
            &u_buf,
        );
        let framebuffers = create_framebuffers(&device, &swapchain, &image_views, render_pass);
        let command_pool = create_command_pool(&device, &physical_device);
        let sem_img_available = create_semaphore(&device);
        let sem_render_finish = create_semaphore(&device);
        let acquire_inflight = create_fence(&device);
        let (vertex_buffer, vb_memory) = allocate_buffer(
            &instance,
            physical_device.device,
            &device,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &VERTICES,
        );
        let (index_buffer, ib_memory) = allocate_buffer(
            &instance,
            physical_device.device,
            &device,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &INDICES,
        );
        let draw_data = DrawData {
            vertex_buffer,
            vb_memory,
            index_buffer,
            ib_memory,
            descriptor_sets,
            vertices: VERTICES.to_vec(),
            indices: INDICES.to_vec(),
        };
        let commands = create_command_buffers(
            &device,
            command_pool,
            &framebuffers,
            render_pass,
            &swapchain,
            (pipeline, pipeline_layout),
            &draw_data,
        );
        VulkanApp {
            _entry: entry,
            instance,
            device,
            physical_device,
            graphics_queue,
            present_queue,
            window,
            surface,
            swapchain,
            image_views,
            render_pass,
            descriptor_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers: commands,
            sem_img_available,
            sem_render_finish,
            acquire_inflight,
            images_inflight: [vk::Fence::null(); MAX_FRAMES_IN_FLIGHT],
            draw_data,
            u_buf,
            u_mem,
            start_time: Instant::now(),
            inflight_frame_no: 0,
            resized: false,
        }
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title(WINDOW_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window")
    }

    fn draw_frame(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(
                    &[self.acquire_inflight[self.inflight_frame_no]],
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for fence");
        }
        let img_idx_u32;
        let img_idx_usz;
        let acquire_result = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.sem_img_available[self.inflight_frame_no],
                vk::Fence::null(),
            )
        };
        if acquire_result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR) {
            self.recreate_swapchain();
            self.resized = false;
            return;
        } else if acquire_result.is_err() {
            panic!("Failed to acquire image");
        } else {
            img_idx_u32 = acquire_result.unwrap().0;
            img_idx_usz = img_idx_u32 as usize;
        }
        let ar = self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32;
        let mvp = update_mvp(self.start_time, ar);
        unsafe {
            let map = self
                .device
                .map_memory(
                    self.u_mem[img_idx_usz],
                    0,
                    std::mem::size_of::<UniformBufferObject>() as u64,
                    Default::default(),
                )
                .expect("Failed to map memory") as *mut UniformBufferObject;
            map.copy_from_nonoverlapping(&mvp, 1);
            self.device.unmap_memory(self.u_mem[img_idx_usz]);
        }
        if self.images_inflight[self.inflight_frame_no] != vk::Fence::null() {
            // this frame is currently in use, wait for it
            unsafe {
                self.device.wait_for_fences(
                    &[self.images_inflight[self.inflight_frame_no]],
                    true,
                    u64::MAX,
                )
            }
            .expect("Failed to wait for fence");
        }
        self.images_inflight[self.inflight_frame_no] =
            self.acquire_inflight[self.inflight_frame_no];
        let wait_sem = [self.sem_img_available[self.inflight_frame_no]];
        let signal_sem = [self.sem_render_finish[self.inflight_frame_no]];
        let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_ci = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_sem.len() as u32,
            p_wait_semaphores: wait_sem.as_ptr(),
            p_wait_dst_stage_mask: wait_mask.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[img_idx_u32 as usize],
            signal_semaphore_count: signal_sem.len() as u32,
            p_signal_semaphores: signal_sem.as_ptr(),
        };
        let swapchains = [self.swapchain.swapchain];
        let present_ci = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: signal_sem.len() as u32,
            p_wait_semaphores: signal_sem.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &img_idx_u32,
            p_results: ptr::null_mut(),
        };
        unsafe {
            self.device
                .reset_fences(&[self.acquire_inflight[self.inflight_frame_no]])
                .expect("Failed to reset fence");
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit_ci],
                    self.acquire_inflight[self.inflight_frame_no],
                )
                .expect("Failed to submit graphicsqueue");
            let present_res = self
                .swapchain
                .loader
                .queue_present(self.present_queue, &present_ci);
            if present_res == Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
                || present_res == Err(vk::Result::SUBOPTIMAL_KHR)
                || self.resized
            {
                self.recreate_swapchain();
                self.resized = false;
                return;
            } else if present_res.is_err() {
                panic!("Failed to present queue")
            }
            self.device
                .queue_wait_idle(self.present_queue)
                .expect("Failed to wait for the queue");
        }
        self.inflight_frame_no = (self.inflight_frame_no + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event,
                window_id: _,
            } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => self.resized = true,
                _ => {}
            },
            Event::MainEventsCleared => self.window.request_redraw(),
            Event::RedrawRequested(_) => self.draw_frame(),
            Event::LoopDestroyed => {
                unsafe { self.device.device_wait_idle() }.expect("Failed to wait idle")
            }
            _ => (),
        })
    }

    fn recreate_swapchain(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait on device");
        }
        self.destroy_swapchain();
        let swapchain_supp =
            query_swapchain_capabilities(&self.surface, self.physical_device.device);
        let swapchain = create_swapchain(
            &self.instance,
            &self.device,
            &swapchain_supp,
            &self.physical_device.queue_family,
            &self.surface,
        );
        let images = unsafe { swapchain.loader.get_swapchain_images(swapchain.swapchain) }
            .expect("Failed to get images");
        let image_views = create_image_views(&self.device, &swapchain, &images);
        let render_pass = create_render_pass(&self.device, &swapchain);
        let descriptor_pool = create_descriptor_pool(&self.device, images.len() as u32);
        let pipeline_layout = create_pipeline_layout(&self.device, &[self.descriptor_layout]);
        let pipeline =
            create_graphics_pipeline(&self.device, &swapchain, render_pass, pipeline_layout);
        let framebuffers = create_framebuffers(&self.device, &swapchain, &image_views, render_pass);
        let (u_buf, u_mem) = allocate_uniform_buffers(
            &self.instance,
            self.physical_device.device,
            &self.device,
            images.len(),
        );
        self.draw_data.descriptor_sets = create_descriptor_sets(
            &self.device,
            images.len() as u32,
            descriptor_pool,
            self.descriptor_layout,
            &u_buf,
        );
        let command_buffers = create_command_buffers(
            &self.device,
            self.command_pool,
            &framebuffers,
            render_pass,
            &swapchain,
            (pipeline, pipeline_layout),
            &self.draw_data,
        );
        self.swapchain = swapchain;
        self.image_views = image_views;
        self.render_pass = render_pass;
        self.descriptor_pool = descriptor_pool;
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;
        self.framebuffers = framebuffers;
        self.command_buffers = command_buffers;
        self.u_buf = u_buf;
        self.u_mem = u_mem;
    }

    fn destroy_swapchain(&self) {
        unsafe {
            self.u_buf
                .iter()
                .for_each(|x| self.device.destroy_buffer(*x, None));
            self.u_mem
                .iter()
                .for_each(|x| self.device.free_memory(*x, None));
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.framebuffers
                .iter()
                .for_each(|fb| self.device.destroy_framebuffer(*fb, None));
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.image_views
                .iter()
                .for_each(|iv| self.device.destroy_image_view(*iv, None));
            self.swapchain.destroy();
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.draw_data.ib_memory, None);
            self.device
                .destroy_buffer(self.draw_data.index_buffer, None);
            self.device.free_memory(self.draw_data.vb_memory, None);
            self.device
                .destroy_buffer(self.draw_data.vertex_buffer, None);
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.sem_render_finish[i], None);
                self.device
                    .destroy_semaphore(self.sem_img_available[i], None);
                self.device.destroy_fence(self.acquire_inflight[i], None);
            }
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.destroy_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
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
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
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

fn pick_physical_device(instance: &ash::Instance, surface: &Surface) -> PDeviceAndQueues {
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
) -> Option<(u32, PDeviceAndQueues)> {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    // let device_features = unsafe { instance.get_physical_device_features(physical_device) };
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
                    PDeviceAndQueues {
                        device: physical_device,
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
    device: &PDeviceAndQueues,
    validations: &ValidationsRequested,
) -> (ash::Device, QueueFamilyIndices) {
    let physical_device = device.device;
    let queue_families = device.queue_family;
    let queue_families_set = queue_families
        .as_array()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut queue_create_infos = Vec::with_capacity(queue_families_set.len());
    let queue_priorities = [1.0];
    for queue_index in queue_families_set {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_index,
            queue_count: queue_priorities.len() as u32,
            p_queue_priorities: queue_priorities.as_ptr(),
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
    device: &ash::Device,
    swapchain: &SwapchainSupport,
    queue_family: &QueueFamilyIndices,
    surface: &Surface,
) -> Swapchain {
    let capabilities = &swapchain.capabilities;
    let format = swapchain
        .formats
        .iter()
        .find(|sf| {
            sf.format == vk::Format::B8G8R8A8_SRGB
                && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| swapchain.formats.first().unwrap());
    let present_mode = *swapchain
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
    let queue_families_indices = [queue_family.graphics_family, queue_family.present_family];
    let (image_sharing_mode, queue_family_index_count, p_queue_family_indices) =
        if queue_family.graphics_family != queue_family.present_family {
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
    Swapchain::new(instance, device, &create_info)
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
            flags: vk::ImageViewCreateFlags::default(),
            image: *image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain.image_format,
            components: vk::ComponentMapping::default(),
            subresource_range,
        };
        retval.push(
            unsafe { device.create_image_view(&create_info, None) }
                .expect("Failed to create Image View"),
        );
    }
    retval
}

fn create_pipeline_layout(
    device: &ash::Device,
    layouts: &[vk::DescriptorSetLayout],
) -> vk::PipelineLayout {
    let layout_ci = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        set_layout_count: layouts.len() as u32,
        p_set_layouts: layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };
    unsafe { device.create_pipeline_layout(&layout_ci, None) }.expect("Failed to create layout")
}

fn create_render_pass(device: &ash::Device, swapchain: &Swapchain) -> vk::RenderPass {
    let color_attachment = [vk::AttachmentDescription {
        flags: Default::default(),
        format: swapchain.image_format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    }];
    let color_attachment_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = [vk::SubpassDescription {
        flags: Default::default(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: color_attachment_ref.len() as u32,
        p_color_attachments: color_attachment_ref.as_ptr(),
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];
    let dependency = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::empty(),
    }];
    let render_pass_ci = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        attachment_count: color_attachment.len() as u32,
        p_attachments: color_attachment.as_ptr(),
        subpass_count: subpass.len() as u32,
        p_subpasses: subpass.as_ptr(),
        dependency_count: dependency.len() as u32,
        p_dependencies: dependency.as_ptr(),
    };
    unsafe { device.create_render_pass(&render_pass_ci, None) }
        .expect("Failed to create render pass")
}

fn create_graphics_pipeline(
    device: &ash::Device,
    swapchain: &Swapchain,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let main_str = CString::new("main").unwrap();
    let vertex_shader = include_bytes!("../target/shaders/triangle.vert.spv");
    let fragment_shader = include_bytes!("../target/shaders/triangle.frag.spv");
    let vertex_module = create_shader_module(device, vertex_shader);
    let fragment_module = create_shader_module(device, fragment_shader);
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::default(),
            stage: vk::ShaderStageFlags::VERTEX,
            module: vertex_module,
            p_name: main_str.as_ptr(),
            p_specialization_info: ptr::null(),
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::default(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fragment_module,
            p_name: main_str.as_ptr(),
            p_specialization_info: ptr::null(),
        },
    ];
    let binding_descriptions = Vertex::binding_descriptions();
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        vertex_binding_description_count: binding_descriptions.len() as u32,
        p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
        vertex_attribute_description_count: attribute_descriptions.len() as u32,
        p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
    };
    let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
    };
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain.extent.width as f32,
        height: swapchain.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain.extent,
    }];
    let viewport_state_ci = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
    };
    let rasterizer_ci = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
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
    let blending_settings = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::all(),
    }];
    let blending_ci = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: blending_settings.len() as u32,
        p_attachments: blending_settings.as_ptr(),
        blend_constants: [0.0; 4],
    };
    let pipeline_ci = vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &vertex_input_ci,
        p_input_assembly_state: &input_assembly_ci,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state_ci,
        p_rasterization_state: &rasterizer_ci,
        p_multisample_state: &multisampling_ci,
        p_depth_stencil_state: ptr::null(),
        p_color_blend_state: &blending_ci,
        p_dynamic_state: ptr::null(),
        layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
    };
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
    }
    .expect("Failed to create graphics pipeline")
    .pop()
    .unwrap();
    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }
    pipeline
}

fn create_shader_module(device: &ash::Device, shader_code: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::default(),
        code_size: shader_code.len(),
        p_code: shader_code.as_ptr() as *const u32,
    };
    unsafe { device.create_shader_module(&create_info, None) }
        .expect("Failed to create shader module")
}

fn create_framebuffers(
    device: &ash::Device,
    swapchain: &Swapchain,
    image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    let mut retval = Vec::with_capacity(image_views.len());
    for view in image_views {
        let attachments = [*view];
        let fb_ci = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: swapchain.extent.width,
            height: swapchain.extent.height,
            layers: 1,
        };
        let fb = unsafe { device.create_framebuffer(&fb_ci, None) }
            .expect("Failed to create frambebuffer");
        retval.push(fb);
    }
    retval
}

fn create_command_pool(
    device: &ash::Device,
    physical_device: &PDeviceAndQueues,
) -> vk::CommandPool {
    let pool_ci = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        queue_family_index: physical_device.queue_family.graphics_family,
    };
    unsafe { device.create_command_pool(&pool_ci, None) }.expect("Failed to create command pool")
}

fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    fbs: &[vk::Framebuffer],
    render_pass: vk::RenderPass,
    swapchain: &Swapchain,
    (pipeline, pipeline_layout): (vk::Pipeline, vk::PipelineLayout),
    data: &DrawData,
) -> Vec<vk::CommandBuffer> {
    let alloc_ci = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: fbs.len() as u32,
    };
    let cmd_buffers = unsafe { device.allocate_command_buffers(&alloc_ci) }
        .expect("Failed to allocate command buffers");
    for i in 0..fbs.len() {
        let command_buffer = cmd_buffers[i];
        let framebuffer = fbs[i];
        let dset = data.descriptor_sets[i];
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            p_inheritance_info: ptr::null(),
        };
        let clear_value = [vk::ClearValue::default()];
        let rp_begin = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass,
            framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            },
            clear_value_count: clear_value.len() as u32,
            p_clear_values: clear_value.as_ptr(),
        };
        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer");
            device.cmd_begin_render_pass(command_buffer, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[data.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                data.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[dset],
                &[],
            );
            device.cmd_draw_indexed(command_buffer, data.indices.len() as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(command_buffer);
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command_buffer");
        }
    }
    cmd_buffers
}

fn create_semaphore(device: &ash::Device) -> [vk::Semaphore; MAX_FRAMES_IN_FLIGHT] {
    let mut retval = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
    for semaphore in retval.iter_mut() {
        let semaphore_ci = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
        };
        *semaphore = unsafe { device.create_semaphore(&semaphore_ci, None) }
            .expect("Failed to create semaphore");
    }
    retval
}

fn create_fence(device: &ash::Device) -> [vk::Fence; MAX_FRAMES_IN_FLIGHT] {
    let mut retval = [vk::Fence::null(); MAX_FRAMES_IN_FLIGHT];
    for fence in retval.iter_mut() {
        let fence_ci = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };
        *fence =
            unsafe { device.create_fence(&fence_ci, None) }.expect("Failed to create semaphore");
    }
    retval
}

fn create_buffer(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_ci = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
    };
    let buffer =
        unsafe { device.create_buffer(&buffer_ci, None) }.expect("Failed to create buffer");
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_alloc_ci = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: mem_requirements.size,
        memory_type_index: find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            properties,
        ),
    };
    let device_memory =
        unsafe { device.allocate_memory(&mem_alloc_ci, None) }.expect("Failed to allocate memory");
    unsafe { device.bind_buffer_memory(buffer, device_memory, 0) }
        .expect("Failed to bind buffer and device memory");
    (buffer, device_memory)
}

fn copy_buffer(
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
) {
    let buf_ci = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };
    let cmd_bufs = unsafe { device.allocate_command_buffers(&buf_ci) }
        .expect("Failed to allocate command buffer");
    let cmd_begin = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: ptr::null(),
    };
    let copy_region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    };
    let submit_ci = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: cmd_bufs.len() as u32,
        p_command_buffers: cmd_bufs.as_ptr(),
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    };
    unsafe {
        let cmd_buf = cmd_bufs[0];
        device
            .begin_command_buffer(cmd_buf, &cmd_begin)
            .expect("Failed to begin command");
        device.cmd_copy_buffer(cmd_buf, src, dst, &[copy_region]);
        device
            .end_command_buffer(cmd_buf)
            .expect("Failed to end command buffer");
        device
            .queue_submit(queue, &[submit_ci], vk::Fence::null())
            .expect("Failed to submit to queue");
        device
            .queue_wait_idle(queue)
            .expect("Failed to wait for queue");
        device.free_command_buffers(pool, &cmd_bufs);
    }
}

fn allocate_buffer<T: Sized>(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    content: vk::BufferUsageFlags,
    data: &[T],
) -> (vk::Buffer, vk::DeviceMemory) {
    // reminder to future myself:
    // for faster performance disable COHERENT and flush manually
    let size = (std::mem::size_of::<T>() * data.len()) as u64;
    let (staging_buf, staging_mem) = create_buffer(
        instance,
        physical_device,
        device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    unsafe {
        let mapped = device
            .map_memory(staging_mem, 0, size, Default::default())
            .expect("Failed to map memory") as *mut T;
        mapped.copy_from_nonoverlapping(data.as_ptr(), data.len());
        device.unmap_memory(staging_mem);
    }
    let (vertex_buf, vertex_mem) = create_buffer(
        instance,
        physical_device,
        device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | content,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    copy_buffer(staging_buf, vertex_buf, size, device, pool, queue);
    unsafe {
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);
    }
    (vertex_buf, vertex_mem)
}

fn allocate_uniform_buffers(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    images_no: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let size = std::mem::size_of::<UniformBufferObject>() as u64;
    let (mut r_buf, mut r_mem) = (Vec::with_capacity(images_no), Vec::with_capacity(images_no));
    for _ in 0..images_no {
        let (c_buf, c_mem) = create_buffer(
            instance,
            physical_device,
            device,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        r_buf.push(c_buf);
        r_mem.push(c_mem);
    }
    (r_buf, r_mem)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    suitable_types: u32,
    suitable_properties: vk::MemoryPropertyFlags,
) -> u32 {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_properties.memory_type_count {
        let is_suitable_type = ((1 << i) & suitable_types) != 0;
        let has_suitable_properties = mem_properties.memory_types[i as usize].property_flags
            & suitable_properties
            == suitable_properties;
        if is_suitable_type && has_suitable_properties {
            return i;
        }
    }
    panic!("Failed to find suitable memory")
}

fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let ubo_l_bind = [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        p_immutable_samplers: ptr::null(),
    }];
    let l_ci = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        binding_count: ubo_l_bind.len() as u32,
        p_bindings: ubo_l_bind.as_ptr(),
    };
    unsafe { device.create_descriptor_set_layout(&l_ci, None) }
        .expect("Failed to create descriptor set layout")
}

fn update_mvp(start_time: Instant, aspect_ratio: f32) -> UniformBufferObject {
    let current_time = Instant::now();
    let elapsed = (current_time - start_time).as_millis() as f32 / 1000.0;
    let model = Matrix4::from_angle_z(Deg(elapsed * 90_f32));
    let view = Matrix4::look_at_rh(
        Point3::new(2_f32, 2.0, 2.0),
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    let proj = perspective(Deg(45_f32), aspect_ratio, 0.1, 10.0);
    UniformBufferObject { model, view, proj }
}

fn create_descriptor_pool(device: &ash::Device, img_no: u32) -> vk::DescriptorPool {
    let psize = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: img_no,
    }];
    let pci = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        max_sets: img_no,
        pool_size_count: psize.len() as u32,
        p_pool_sizes: psize.as_ptr(),
    };
    unsafe { device.create_descriptor_pool(&pci, None) }.expect("Failed to create descriptor pool")
}

fn create_descriptor_sets(
    device: &ash::Device,
    img_no: u32,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    bufs: &[vk::Buffer],
) -> Vec<vk::DescriptorSet> {
    let layouts = vec![layout; img_no as usize];
    let ci = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool: pool,
        descriptor_set_count: img_no,
        p_set_layouts: layouts.as_ptr(),
    };
    let sets =
        unsafe { device.allocate_descriptor_sets(&ci) }.expect("Failed to allocate descriptor set");
    for (i, descriptor) in sets.iter().enumerate() {
        let bci = [vk::DescriptorBufferInfo {
            buffer: bufs[i],
            offset: 0,
            range: std::mem::size_of::<UniformBufferObject>() as u64,
        }];
        let write_ci = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: *descriptor,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: bci.len() as u32,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: bci.as_ptr(),
            p_texel_buffer_view: ptr::null(),
        };
        unsafe { device.update_descriptor_sets(&[write_ci], &[]) };
    }
    sets
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);
    let app = VulkanApp::new(window);
    app.main_loop(event_loop);
}
