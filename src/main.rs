use ash::vk;
use std::ffi::CStr;
use std::ptr;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const WINDOW_NAME: &str = "Vulkan test";
const ENGINE_NAME: &str = "No engine";

struct VulkanApp;

impl VulkanApp {
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

// fn create_instance() -> vk::Instance {
//     let info = vk::ApplicationInfo {
//         s_type: vk::StructureType::APPLICATION_INFO,
//         p_next: ptr::null(),
//         p_application_name: CStr:
//         application_version: vk::make_version(0, 1, 0),
//         p_engine_name: ENGINE_NAME.as_ptr(),
//         engine_version: vk::make_version(0, 1, 0),
//         api_version: vk::API_VERSION_1_2,
//     };
//     todo!()
// }

fn main() {
    let event_loop = EventLoop::new();
    VulkanApp::init_window(&event_loop);
    VulkanApp::main_loop(event_loop);
}
