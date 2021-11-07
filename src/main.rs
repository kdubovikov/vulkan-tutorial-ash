mod utils;

use std::{ffi::{CStr, CString, c_void}, os::raw::c_char, ptr};

use ash::vk;

use log::Log;
use winit::{event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

use utils::platform::{create_surface, required_extension_names};
use utils::debug::{setup_debug_utils, ValidationInfo, populate_debug_messenger_create_info};
use utils::tools::{vk_to_string};

const WINDOW_NAME: &'static str = "Vulkan Tutorial";

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SurfaceStuff {
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR
}

struct VulkanApp {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    _physical_device: vk::PhysicalDevice,
    _device: ash::Device,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
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

trait PhysicalDeviceFactory {
    fn pick_physical_device(instance: &ash::Instance, surface_stuff: &SurfaceStuff) -> vk::PhysicalDevice;
    fn is_physical_device_suitable(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> bool;
    fn find_queue_family(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> QueueFamilyIndices;
}

trait LogicalDeviceFactory {
    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, validation_info: &ValidationInfo, surface_stuff: &SurfaceStuff) -> (ash::Device, vk::Queue, vk::Queue);
}

trait SurfaceFactory {
    fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &winit::window::Window) -> SurfaceStuff;
}

impl SurfaceFactory for VulkanApp {
    fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> SurfaceStuff {
        let surface = unsafe {
            create_surface(entry, instance, window).expect("Error creating surface")
        };

        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        SurfaceStuff {
            surface,
            surface_loader,
        }
    }
}

impl LogicalDeviceFactory for VulkanApp {
    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, validation: &ValidationInfo, surface_stuff: &SurfaceStuff) -> (ash::Device, vk::Queue, vk::Queue) {
        let indices = VulkanApp::find_queue_family(instance, physical_device, &surface_stuff);

        let queue_properties = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: indices.graphics_family.unwrap(),
            p_queue_priorities: queue_properties.as_ptr(),
            queue_count: queue_properties.len() as u32,
        };

        let physical_device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let required_validation_layer_raw_names: Vec<CString> = validation
            .required_validation_layers
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect();
        let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let enabled_extension_raw_names: Vec<CString> = vec![CString::new("VK_KHR_portability_subset").unwrap()];
        let enabled_extension_names: Vec<*const c_char> = enabled_extension_raw_names.iter().map(|name| name.as_ptr()).collect();

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: if validation.enabled {
                enable_layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names: if validation.enabled {
                enable_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_extension_count: enabled_extension_names.len() as u32,
            pp_enabled_extension_names: enabled_extension_names.as_ptr(),
            // enabled_extension_count: 0,
            // pp_enabled_extension_names: ptr::null(),
            p_enabled_features: &physical_device_features,
        };
        
        let device: ash::Device = unsafe {
            instance.create_device(*physical_device, &device_create_info, None)
                .expect("Failed to create logical device")
        };
        let graphics_queue = unsafe {
            device.get_device_queue(indices.graphics_family.unwrap(), 0)
        };

        let present_queue = unsafe {
            device.get_device_queue(indices.present_family.unwrap(), 0)
        };

        (device, graphics_queue, present_queue)
    }
}

impl PhysicalDeviceFactory for VulkanApp {
    fn pick_physical_device(instance: &ash::Instance, surface_stuff: &SurfaceStuff) -> vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        println!("{} devices (GPU) found with vulkan support", physical_devices.len());

        let mut result = None;

        for physical_device in physical_devices.iter() {
            if Self::is_physical_device_suitable(instance, physical_device, surface_stuff) {
                if result.is_none() {
                    result = Some(physical_device);
                }
            }
        }

        *result.unwrap()
    }

    fn is_physical_device_suitable(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> bool {
        let device_properties = unsafe {
            instance.get_physical_device_properties(*physical_device)
        };

        let device_features = unsafe {
            instance.get_physical_device_features(*physical_device)
        };

        let device_queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(*physical_device)
        };

        let device_type = match device_properties.device_type {
            vk::PhysicalDeviceType::CPU => "CPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "INTEGRATED_GPU",
            vk::PhysicalDeviceType::DISCRETE_GPU => "DISCRETE_GPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "VIRTUAL_GPU",
            vk::PhysicalDeviceType::OTHER => "OTHER",
            _ => panic!(),
        };

        let device_name = vk_to_string(&device_properties.device_name);
        println!("Device {}, id: {}, type: {}", device_name, device_properties.device_id, device_type);

        let major_version = vk::api_version_major(device_properties.api_version);
        let minor_version = vk::api_version_minor(device_properties.api_version);
        let patch_version = vk::api_version_patch(device_properties.api_version);

        println!(
            "\tAPI Version: {}.{}.{}",
            major_version, minor_version, patch_version
        );

        println!("\tSupport Queue Family: {}", device_queue_families.len());
        println!("\t\tQueue Count | Graphics, Compute, Transfer, Sparse Binding");
        
        for queue_family in device_queue_families.iter() {
            let is_graphics_support = if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                "support"
            } else {
                "unsupport"
            };
            let is_compute_support = if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                "support"
            } else {
                "unsupport"
            };
            let is_transfer_support = if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            {
                "support"
            } else {
                "unsupport"
            };
            let is_sparse_support = if queue_family
                .queue_flags
                .contains(vk::QueueFlags::SPARSE_BINDING)
            {
                "support"
            } else {
                "unsupport"
            };

            println!(
                "\t\t{}\t    | {},  {},  {},  {}",
                queue_family.queue_count,
                is_graphics_support,
                is_compute_support,
                is_transfer_support,
                is_sparse_support
            );
        }

        println!("Gemoetry shader support: {}", 
        if device_features.geometry_shader == 1 {
            "Supports"
        } else {
            "Does not support"
        });

        let indices = Self::find_queue_family(instance, physical_device, surface_stuff);
        return indices.is_complete();
    }

    fn find_queue_family(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> QueueFamilyIndices {
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(*physical_device)
        };

        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None
        };

        let mut index = 0;
        for queue_family in queue_families.iter() {
            if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index);
            }

            let is_present_support = unsafe {
                surface_stuff
                    .surface_loader
                    .get_physical_device_surface_support(*physical_device, index as u32, surface_stuff.surface)
            };

            if let Ok(true) = is_present_support {
                if queue_family.queue_count > 0 {
                    queue_family_indices.present_family = Some(index);
                }
            }

            if queue_family_indices.is_complete() {
                break;
            }

            index += 1;
        }

        queue_family_indices
    }
}

impl VulkanApp {
    pub fn new(window: &winit::window::Window) -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.unwrap();
        let instance = VulkanApp::create_instance(&entry);
        let surface_stuff = VulkanApp::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_messenger) = setup_debug_utils(VALIDATION.enabled, &entry, &instance);
        let physical_device = VulkanApp::pick_physical_device(&instance, &surface_stuff);
        let (device, graphics_queue, present_queue) = VulkanApp::create_logical_device(&instance, &physical_device, &VALIDATION, &surface_stuff);

        VulkanApp {
            _entry: entry,
            instance: instance,
            debug_utils_loader: debug_utils_loader,
            debug_messenger: debug_messenger,
            _physical_device: physical_device,
            _device: device,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            graphics_queue,
            present_queue
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
            self._device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

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

    let mut vulkan_app = VulkanApp::new(&window); 
    vulkan_app.main_loop(event_loop, window);
}
