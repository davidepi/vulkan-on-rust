use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CString;
use std::ptr;
use validation::{check_validation_layers_support, VALIDATION_ON};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
mod platform;
mod validation;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const WINDOW_NAME: &str = "Vulkan test";
const ENGINE_NAME: &str = "No engine";

const REQUIRED_VALIDATIONS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl VulkanApp {
    pub fn new() -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.expect("Could not load Vulkan library");
        let instance = create_instance(&entry);
        VulkanApp { entry, instance }
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
            .with_title(WINDOW_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window")
    }

    pub fn main_loop(event_loop: EventLoop<()>) -> ! {
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
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn create_instance(entry: &ash::Entry) -> ash::Instance {
    if VALIDATION_ON && !check_validation_layers_support(entry, &REQUIRED_VALIDATIONS[..]) {
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
    let required_extensions = platform::winit_get_required_extension_names();
    let requested_val_layers_names = REQUIRED_VALIDATIONS
        .iter()
        .map(|x| CString::new(*x).unwrap())
        .collect::<Vec<_>>();
    let requested_val_layers = requested_val_layers_names
        .iter()
        .map(|x| x.as_ptr())
        .collect::<Vec<_>>();
    let creation_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: if VALIDATION_ON {
            requested_val_layers.len() as u32
        } else {
            0
        },
        pp_enabled_layer_names: if VALIDATION_ON {
            requested_val_layers.as_ptr()
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

fn main() {
    let event_loop = EventLoop::new();
    VulkanApp::init_window(&event_loop);
    let instance = VulkanApp::new();
    VulkanApp::main_loop(event_loop);
}
