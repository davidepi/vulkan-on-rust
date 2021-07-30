use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use platform::create_surface;
use std::collections::BTreeSet;
use std::ffi::CString;
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

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: Surface,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.expect("Could not load Vulkan library");
        let validations = ValidationsRequested::new(&REQUIRED_VALIDATIONS[..]);
        let instance = create_instance(&entry, &validations);
        let surface = Surface::new(&entry, &instance, window);
        let physical_device = pick_physical_device(&instance, &surface);
        let (device, queue_indices) =
            create_logical_device(&instance, physical_device, &validations, &surface);
        let graphics_queue = unsafe { device.get_device_queue(queue_indices.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_indices.present_family, 0) };
        VulkanApp {
            entry,
            instance,
            device,
            graphics_queue,
            present_queue,
            surface,
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

fn pick_physical_device(instance: &ash::Instance, surface: &Surface) -> vk::PhysicalDevice {
    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
    log::info!("Found {} GPUs in the system", physical_devices.len());
    physical_devices
        .into_iter()
        .map(|device| {
            (
                device,
                rate_physical_device_suitability(instance, device, surface),
            )
        })
        .filter(|(_, score)| *score != 0)
        .max_by_key(|(_, score)| *score)
        .expect("No compatible physical devices found")
        .0
}

fn rate_physical_device_suitability(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface: &Surface,
) -> u32 {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };
    let device_features = unsafe { instance.get_physical_device_features(device) };
    let mut score = match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
        vk::PhysicalDeviceType::CPU => 1,
        vk::PhysicalDeviceType::OTHER => 10,
        _ => 10,
    };
    let queues = find_queue_families(instance, device, surface);
    if device_features.geometry_shader == 0 || !queues.is_complete() {
        score = 0;
    }
    score
}

fn find_queue_families(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface: &Surface,
) -> QueueFamilyIndicesIncomplete {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let graphics_family = queue_families
        .iter()
        .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    let present_family = (0..queue_families.len()).into_iter().find(|x| unsafe {
        surface
            .loader
            .get_physical_device_surface_support(device, *x as u32, surface.khr())
            .unwrap()
    });
    QueueFamilyIndicesIncomplete {
        graphics_family,
        present_family,
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    validations: &ValidationsRequested,
    surface: &Surface,
) -> (ash::Device, QueueFamilyIndices) {
    let queue_families = find_queue_families(instance, physical_device, surface)
        .get_complete()
        .unwrap();
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
        enabled_extension_count: 0,
        pp_enabled_extension_names: ptr::null(),
        p_enabled_features: &physical_features,
    };
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("Failed to create logical device");
    (device, queue_families)
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);
    let app = VulkanApp::new(&window);
    app.main_loop(event_loop);
}
