mod utils;

use std::{ffi::{CString, c_void}, ptr};

use ash::vk;

use winit::{event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

use utils::platform::required_extension_names;
use utils::debug::{setup_debug_utils, ValidationInfo, populate_debug_messenger_create_info};

const WINDOW_NAME: &'static str = "Vulkan Tutorial";

struct VulkanApp {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

const VALIDATION: ValidationInfo = ValidationInfo {
    enabled: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

trait WindowedApp {
    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window;
    fn process_key_input(control_flow: &mut ControlFlow, input: KeyboardInput);
}

impl WindowedApp for VulkanApp {
    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .build(event_loop)
            .expect("Failed to create window")

    }

    fn process_key_input(control_flow: &mut ControlFlow, input: KeyboardInput) {
        match input {
            KeyboardInput { virtual_keycode, state, .. } => {
                match (virtual_keycode, state) {
                    (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                        dbg!();
                        *control_flow = ControlFlow::Exit;
                    },
                    _ => {}
                }
            },  
        }
    }
}

impl VulkanApp {
    pub fn new() -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.unwrap();
        let instance = VulkanApp::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = setup_debug_utils(VALIDATION.enabled, &entry, &instance);
        
        VulkanApp {
            _entry: entry,
            instance: instance,
            debug_utils_loader: debug_utils_loader,
            debug_messenger: debug_messenger
        }
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let app_name = CString::new(WINDOW_NAME).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        const APPLICATION_VERSION: u32 = 1;
        const ENGINE_VERSION: u32 = 1;
        const API_VERSION: u32 = 1;

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: ENGINE_VERSION,
            api_version: vk::make_api_version(0, 1, 0, 0)
        };

        let debug_utils_create_info = populate_debug_messenger_create_info();
        let extension_names = required_extension_names();

        let required_validation_layer_raw_names: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        
        let enable_layer_names: Vec<*const i8> = required_validation_layer_raw_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let create_info = ash::vk::InstanceCreateInfo {
            s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: if VALIDATION.enabled {
                &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
            } else {
                ptr::null()
            },
            flags: ash::vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            pp_enabled_extension_names: extension_names.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            enabled_layer_count: if VALIDATION.enabled {
                enable_layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names:  if VALIDATION.enabled {
                enable_layer_names.as_ptr()
            } else {
                ptr::null()
            },
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
    }

    fn draw_frame(&mut self) {
    }

    fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        },
                        WindowEvent::KeyboardInput { input, .. } => {
                            Self::process_key_input(control_flow, input)
                        },
                        _ => {},
                    }
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                },
                Event::RedrawRequested(_window_id) => {
                    self.draw_frame();
                },
                _ => {}
            }
        })
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION.enabled {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);

    let mut vulkan_app = VulkanApp::new(); 
    vulkan_app.main_loop(event_loop, window);
}
