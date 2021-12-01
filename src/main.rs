mod utils;

use std::{ffi::{CStr, CString, c_void}, fs::File, os::raw::c_char, path::Path, ptr, io::Read};

use ash::vk::{self, PresentModeKHR, SurfaceFormatKHR};

use log::Log;
use winit::{event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

use utils::platform::{create_surface, required_extension_names};
use utils::debug::{setup_debug_utils, ValidationInfo, populate_debug_messenger_create_info};
use utils::tools::{vk_to_string};
use num::clamp;

const WINDOW_NAME: &'static str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SurfaceStuff {
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR
}

struct SwapChainStuff {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D
}

struct SwapChainSupportDetail {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>
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
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    _swapchain_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    swapchain_imageview: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout
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
    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, validation_info: &ValidationInfo, surface_stuff: &SurfaceStuff) -> (ash::Device, vk::Queue, vk::Queue, QueueFamilyIndices);
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
    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, validation: &ValidationInfo, surface_stuff: &SurfaceStuff) -> (ash::Device, vk::Queue, vk::Queue, QueueFamilyIndices) {
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

            
        let enabled_extension_raw_names: Vec<CString> = vec![
            CString::new("VK_KHR_portability_subset").unwrap(), 
            CString::new(ash::extensions::khr::Swapchain::name().to_bytes()).unwrap()
        ];
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

        (device, graphics_queue, present_queue, indices)
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
        let (device, graphics_queue, present_queue, family_indices) = VulkanApp::create_logical_device(&instance, &physical_device, &VALIDATION, &surface_stuff);


        let swapchain_stuff = VulkanApp::create_swapchain(
            &instance,
            &device,
            physical_device,
            &surface_stuff,
            &family_indices,
        );

        let swapchain_imageview = VulkanApp::create_image_views(&device, swapchain_stuff.format, &swapchain_stuff.images);
        let pipeline_layout = VulkanApp::create_graphics_pipeline(&device, swapchain_stuff.extent);

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
            present_queue,
            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            _swapchain_format: swapchain_stuff.format,
            _swapchain_images: swapchain_stuff.images,
            _swapchain_extent: swapchain_stuff.extent,
            swapchain_imageview,
            pipeline_layout
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

    fn choose_swapchain_format(available_formats: &Vec<SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for available_format in available_formats {
            if available_format.format == vk::Format::R8G8B8A8_SRGB && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return available_format.clone();
            }
        }

        return available_formats.first().unwrap().clone();
    }

    fn choose_swapchain_present_mode(available_modes: &Vec<PresentModeKHR>) -> vk::PresentModeKHR {
        for &available_mode in available_modes {
            if available_mode == vk::PresentModeKHR::MAILBOX {
                return available_mode;
            }
        }

        vk::PresentModeKHR::FIFO
    }

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width == u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: clamp(
                    WINDOW_WIDTH,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width
                ),
                height: clamp(
                    WINDOW_HEIGHT,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height
                ),
            }
        }
    }
    fn query_swapchain_support(physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> SwapChainSupportDetail {
        unsafe {
            let capabilities = surface_stuff
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface capabilities.");

            let formats = surface_stuff
                .surface_loader
                .get_physical_device_surface_formats(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface formats.");

            let present_modes = surface_stuff
                .surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface_stuff.surface)
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetail {
                capabilities,
                formats,
                present_modes
            }
        }
    }
    fn create_swapchain(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff, queue_family: &QueueFamilyIndices) -> SwapChainStuff {
        let swapchain_support = VulkanApp::query_swapchain_support(physical_device, surface_stuff);
        let surface_format = VulkanApp::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = VulkanApp::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = VulkanApp::choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) = 
            if queue_family.graphics_family != queue_family.present_family {
                (
                    vk::SharingMode::EXCLUSIVE,
                    2,
                    vec![
                        queue_family.graphics_family.unwrap(),
                        queue_family.present_family.unwrap()
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface_stuff.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
        };

        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create Swapchain!")
        };

        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get Swapchain Images.")
        };

        SwapChainStuff {
            swapchain_loader,
            swapchain,
            format: surface_format.format,
            extent: extent,
            images: swapchain_images,
        }
    }

    fn create_image_views(device: &ash::Device, surface_format: vk::Format, images: &Vec<vk::Image>) -> Vec<vk::ImageView> {
        let mut swapchain_imageviews = vec![];

        for &image in images.iter() {
            let imageview_create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                view_type: vk::ImageViewType::TYPE_2D,
                format: surface_format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange { 
                    aspect_mask: vk::ImageAspectFlags::COLOR, 
                    base_mip_level: 0, 
                    level_count: 1, 
                    base_array_layer: 0, 
                    layer_count: 1 
                },
                image
            };

            let imageview = unsafe {
                device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create Image View!")
            };

            swapchain_imageviews.push(imageview);
        }

        swapchain_imageviews
    }

    fn create_graphics_pipeline(device: &ash::Device, swapchain_extent: vk::Extent2D) -> vk::PipelineLayout {
        let vert_shader_code = VulkanApp::read_shader_code(Path::new("shaders/spv/09-shader-base.vert.spv"));
        let frag_shader_code = VulkanApp::read_shader_code(Path::new("shaders/spv/09-shader-base.frag.spv"));
        let vert_shader_module = VulkanApp::create_shader_module(device, vert_shader_code);
        let frag_shader_module = VulkanApp::create_shader_module(device, frag_shader_code);

        let main_function_name = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::VERTEX
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::FRAGMENT
            },
        ];

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
        };

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            p_next: ptr::null(),
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        };

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr()
        };

        let _rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: vk::FALSE,
            depth_bias_slope_factor: 0.0,
        };
        let _multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            p_next: ptr::null(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: vk::FALSE,
            alpha_to_coverage_enable: vk::FALSE,
        };

        let stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };

        let _depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        }];

        let _color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        pipeline_layout
    }

    fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
        };

        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn read_shader_code(shader_path: &Path) -> Vec<u8> {
        let spv_file = File::open(shader_path)
            .expect(&format!("Failed to find spv file at {:?}", shader_path));
        let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

        bytes_code
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
            self._device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            for &imageview in self.swapchain_imageview.iter() {
                self._device.destroy_image_view(imageview, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
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
