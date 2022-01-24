mod utils;
mod vertex;

use std::{ffi::{CStr, CString, c_void}, fs::File, os::raw::c_char, path::Path, ptr, io::Read};

use ash::{vk::{self, PresentModeKHR, SurfaceFormatKHR}, prelude::VkResult};

use cgmath::{Matrix, Matrix4, SquareMatrix, Deg, Point3, Vector3};
use image::GenericImageView;
use log::Log;
use vertex::{Vertex};
use winit::{event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

use utils::platform::{create_surface, required_extension_names};
use utils::debug::{setup_debug_utils, ValidationInfo, populate_debug_messenger_create_info};
use utils::tools::{vk_to_string};
use utils::shader::{read_shader_code, create_shader_module};
use num::clamp;

const WINDOW_NAME: &'static str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

const TEXTURE_PATH: &'static str = "assets/viking_room.png";
const MODEL_PATH: &'static str = "assets/viking_room.obj";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>
}

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

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageview: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    uniform_transform: UniformBufferObject,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    family_indices: QueueFamilyIndices,
    window: winit::window::Window,
    is_framebuffer_resized: bool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_set_layout: vk::DescriptorSetLayout,

    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_sampler: vk::Sampler,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    indices_count: u32
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

trait SwapchainFactory {
    fn choose_swapchain_format(available_formats: &Vec<SurfaceFormatKHR>) -> vk::SurfaceFormatKHR;
    fn choose_swapchain_present_mode(available_modes: &Vec<PresentModeKHR>) -> vk::PresentModeKHR; 
    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D;
    fn query_swapchain_support(physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> SwapChainSupportDetail; 
    fn create_swapchain(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff, queue_family: &QueueFamilyIndices) -> SwapChainStuff;
    fn recreate_swapchain(&mut self);
    fn cleanup_swapchain(&mut self); 
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

        let physical_device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

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

        let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

        return indices.is_complete() && is_support_sampler_anisotropy;
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

impl SwapchainFactory for VulkanApp {
    fn choose_swapchain_format(available_formats: &Vec<SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for available_format in available_formats {
            if available_format.format == vk::Format::R8G8B8A8_UNORM && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
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

    fn recreate_swapchain(&mut self) {
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
        };

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        let swapchain_stuff = VulkanApp::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &surface_suff,
            &self.family_indices,
        );

        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.images;
        self.swapchain_format = swapchain_stuff.format;
        self.swapchain_extent = swapchain_stuff.extent;

        self.swapchain_imageview = VulkanApp::create_image_views(
            &self.device,
            self.swapchain_format,
            &self.swapchain_images,
        );

        self.render_pass = VulkanApp::create_render_pass(&self.instance, &self.device, self.physical_device, self.swapchain_format);
        let (graphics_pipeline, pipeline_layout) = VulkanApp::create_graphics_pipeline(
            &self.device,
            swapchain_stuff.extent,
            self.render_pass,
            self.descriptor_set_layout
        );
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        self.swapchain_framebuffers = VulkanApp::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageview,
            self.depth_image_view,
            self.swapchain_extent,
        );
        self.command_buffers = VulkanApp::create_command_buffers(
            &self.device,
            self.command_pool,
            self.graphics_pipeline,
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_extent,
            self.vertex_buffer,
            self.index_buffer,
            self.pipeline_layout,
            &self.descriptor_sets,
            self.indices_count
        );
    }
    
    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_imageview.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
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
}

impl VulkanApp {
    pub fn new(window: winit::window::Window) -> VulkanApp {
        let entry = unsafe { ash::Entry::new() }.unwrap();
        let instance = VulkanApp::create_instance(&entry);
        let surface_stuff = VulkanApp::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_messenger) = setup_debug_utils(VALIDATION.enabled, &entry, &instance);

        let physical_device = VulkanApp::pick_physical_device(&instance, &surface_stuff);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let (device, graphics_queue, present_queue, family_indices) = VulkanApp::create_logical_device(&instance, &physical_device, &VALIDATION, &surface_stuff);

        let command_pool = VulkanApp::create_command_pool(&device, &family_indices);

        let swapchain_stuff = VulkanApp::create_swapchain(
            &instance,
            &device,
            physical_device,
            &surface_stuff,
            &family_indices,
        );

        let (depth_image, depth_image_view, depth_image_memory) = VulkanApp::create_depth_resources(&instance, &device, physical_device, command_pool, graphics_queue, swapchain_stuff.extent, &physical_device_memory_properties);
        let swapchain_imageview = VulkanApp::create_image_views(&device, swapchain_stuff.format, &swapchain_stuff.images);
        let render_pass = VulkanApp::create_render_pass(&instance, &device, physical_device, swapchain_stuff.format);
        let swapchain_framebuffers = VulkanApp::create_framebuffers(&device, render_pass, &swapchain_imageview, depth_image_view, swapchain_stuff.extent);

        let sync_objects = VulkanApp::create_sync_objects(&device);

        let (vertices, indices) = VulkanApp::load_model(&Path::new(MODEL_PATH));
        println!("Loaded model");
        let (vertex_buffer, vertex_buffer_memory) = VulkanApp::create_vertex_buffer(&instance, &device, physical_device, command_pool, graphics_queue, &vertices);
        println!("Created vertex buffer");
        let (index_buffer, index_buffer_memory) = VulkanApp::create_index_buffer(&instance, &device, physical_device, command_pool, graphics_queue, &indices);
        println!("Created index buffer");

        VulkanApp::check_mipmap_support(&instance, physical_device, vk::Format::R8G8B8A8_UNORM);
        let (texture_image, texture_image_memory, mip_levels) = VulkanApp::create_texture_image(
            &device,
            command_pool,
            graphics_queue,
            &physical_device_memory_properties,
            &Path::new(TEXTURE_PATH),
        );

        let texture_sampler = VulkanApp::create_texture_sampler(&device, mip_levels);
        println!("Created texture. mip_levels is {}", mip_levels);
        let texture_image_view = VulkanApp::create_texture_image_view(&device, texture_image, mip_levels);

        let (uniform_buffers, uniform_buffers_memory) = VulkanApp::creaet_uniform_buffers(&instance, &device, physical_device, swapchain_stuff.images.len());
        let ubo_layout = VulkanApp::create_descriptor_set_layout(&device);
        
        let descriptor_pool = VulkanApp::create_descriptor_pool(&device, swapchain_stuff.images.len());
        let descriptor_set_layout = VulkanApp::create_descriptor_set_layout(&device);
        let descriptor_sets = VulkanApp::create_descriptor_sets(&device, descriptor_pool, descriptor_set_layout, &uniform_buffers, swapchain_stuff.images.len(), texture_sampler, texture_image_view);

        let (graphics_pipeline, pipeline_layout ) = VulkanApp::create_graphics_pipeline(&device, swapchain_stuff.extent, render_pass, descriptor_set_layout);
        let command_buffers = VulkanApp::create_command_buffers(
            &device, 
            command_pool, 
            graphics_pipeline, 
            &swapchain_framebuffers, 
            render_pass, 
            swapchain_stuff.extent,
            vertex_buffer, 
            index_buffer,
            pipeline_layout,
            &descriptor_sets,
            indices.len() as u32
        );

        println!("Created all resources");

        VulkanApp {
            entry,
            instance: instance,
            debug_utils_loader: debug_utils_loader,
            debug_messenger: debug_messenger,
            physical_device,
            device,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            graphics_queue,
            present_queue,
            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_format: swapchain_stuff.format,
            swapchain_images: swapchain_stuff.images,
            swapchain_extent: swapchain_stuff.extent,
            swapchain_imageview,
            swapchain_framebuffers,
            pipeline_layout,
            ubo_layout,
            render_pass,
            command_pool,
            command_buffers,

            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffers_memory,

            uniform_transform: UniformBufferObject {
                model: Matrix4::identity(),
                view: Matrix4::look_at_rh(
                    Point3::new(2.0, 2.0, 2.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, -1.0),
                ),
                proj: {
                    let mut proj = cgmath::perspective(
                        Deg(45.0),
                        swapchain_stuff.extent.width as f32
                            / swapchain_stuff.extent.height as f32,
                        0.1,
                        10.0,
                        );
                    // proj[1][1] = proj[1][1] * -1.0;
                    proj
                }
            },

            image_available_semaphores: sync_objects.image_available_semaphores,
            render_finished_semaphores: sync_objects.render_finished_semaphores,
            in_flight_fences: sync_objects.inflight_fences,
            current_frame: 0,
            family_indices,
            window: window,
            graphics_pipeline,
            is_framebuffer_resized: false,
            descriptor_sets,
            descriptor_set_layout,

            texture_image,
            texture_image_memory,
            texture_sampler,

            depth_image,
            depth_image_view,
            depth_image_memory,

            indices_count: indices.len() as u32
        }
    }

    fn load_model(path: &Path) -> (Vec<Vertex>, Vec<u32>) {
        let model_obj = tobj::load_obj(path, &tobj::LoadOptions::default())
            .expect("Failed to load model object!");
        let mut vertices = vec![];
        let mut indices = vec![];

        let (models, _) = model_obj;

        for model in models.iter() {
            let mesh = &model.mesh;

            if mesh.texcoords.len() == 0 {
                panic!("Missing texture coordinate for the model.")
            }

            let total_vert_count = mesh.positions.len() / 3;

            for i in 0..total_vert_count {
                let vertex = Vertex {
                    pos: [
                        mesh.positions[i * 3],
                        mesh.positions[i * 3 + 1],
                        mesh.positions[i * 3 + 2]
                    ],
                    color: [1.0, 1.0, 1.0],
                    tex_coord: [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]],
                };

                vertices.push(vertex);
            }

            indices = mesh.indices.clone();
        }

        (vertices, indices)
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

    fn create_graphics_pipeline(device: &ash::Device, swapchain_extent: vk::Extent2D, render_pass: vk::RenderPass, ubo_set_layout: vk::DescriptorSetLayout) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_shader_code = read_shader_code(Path::new("shaders/spv/26-shader-depth.vert.spv"));
        let frag_shader_code = read_shader_code(Path::new("shaders/spv/25-shader-textures.frag.spv"));
        let vert_shader_module = create_shader_module(device, vert_shader_code);
        let frag_shader_module = create_shader_module(device, frag_shader_code);

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

        let binding_info = Vertex::get_binding_descriptions();
        let attribute_info = Vertex::get_attribute_descriptions();

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: attribute_info.len() as u32,
            p_vertex_attribute_descriptions: attribute_info.as_ptr(),
            vertex_binding_description_count: binding_info.len() as u32,
            p_vertex_binding_descriptions: binding_info.as_ptr(),
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

        let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo {
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
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
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

        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
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

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };

        let set_layouts = [ubo_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterization_statue_create_info,
            p_multisample_state: &multisample_state_create_info,
            p_depth_stencil_state: &depth_state_create_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &graphic_pipeline_create_infos,
                    None,
                )
                .expect("Failed to create Graphics Pipeline!.")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (graphics_pipelines[0], pipeline_layout)
    }

    fn create_framebuffers(
        device: &ash::Device, 
        render_pass: vk::RenderPass, 
        image_views: &Vec<vk::ImageView>, 
        depth_image_view: vk::ImageView,
        swapchain_extent: vk::Extent2D
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];

        for &image_view in image_views.iter() {
            let attachments = [image_view, depth_image_view];
            
            let framebuffer_create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width ,
                height: swapchain_extent.height,
                layers: 1,
            };

            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            };

            framebuffers.push(framebuffer);
        }

        framebuffers
    }

    fn create_render_pass(
        instance: &ash::Instance,
        device: &ash::Device,
        physcial_device: vk::PhysicalDevice,
        surface_format: vk::Format
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR
        };

        let depth_attachement = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: VulkanApp::find_depth_format(instance, physcial_device),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachement_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass_description = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: &depth_attachement_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        };

        let render_pass_attachments = [color_attachment, depth_attachement];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];

        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            flags: vk::RenderPassCreateFlags::empty(),
            p_next: ptr::null(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass_description,
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr()
        };

        unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create render pass!")
        }
    }

    fn create_command_pool(device: &ash::Device, queue_families: &QueueFamilyIndices) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index: queue_families.graphics_family.unwrap(),
        };

        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        }
    }
    
    fn create_command_buffers(
        device: &ash::Device, 
        command_pool: vk::CommandPool, 
        graphics_pipeline: vk::Pipeline, 
        framebuffers: &Vec<vk::Framebuffer>,
        render_pass: vk::RenderPass,
        surface_extent: vk::Extent2D,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &Vec<vk::DescriptorSet>,
        indices_count: u32
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: framebuffers.len() as u32,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let commmad_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                p_inheritance_info: ptr::null()
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &commmad_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                }
            },
            vk::ClearValue {
                // clear value for depth buffer
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                }
            }
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };

            let vertex_buffers = [vertex_buffer];
            let offsets = [0_u64];
            let descriptor_sets_to_bind = [descriptor_sets[i]];

            unsafe {
                device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);
                device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &descriptor_sets_to_bind, &[]);

                device.cmd_draw_indexed(command_buffer, indices_count, 1, 0, 0, 0);
                device.cmd_end_render_pass(command_buffer);
                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }

    fn create_sync_objects(device: &ash::Device) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            inflight_fences: vec![],
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };

        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_abailable_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let inflight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create fence!");

                sync_objects.image_available_semaphores.push(image_abailable_semaphore);
                sync_objects.render_finished_semaphores.push(render_finished_semaphore);
                sync_objects.inflight_fences.push(inflight_fence);
            }
        }

        sync_objects
    } 

    fn create_vertex_buffer(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, command_pool: vk::CommandPool, submit_queue: vk::Queue, vertices: &[Vertex]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(vertices) as vk::DeviceSize;

        let mem_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let (staging_buffer, staging_buffer_memory) = VulkanApp::create_buffer(
            device, 
            buffer_size, 
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, 
            &mem_properties
        ); 

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to Map Memory") as *mut Vertex;

            data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = VulkanApp::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &mem_properties,
        );

        VulkanApp::copy_buffer(
            device,
            submit_queue,
            command_pool,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, command_pool: vk::CommandPool, submit_queue: vk::Queue, indices: &[u32]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(indices) as vk::DeviceSize;

        let mem_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let (staging_buffer, staging_buffer_memory) = VulkanApp::create_buffer(
            device, 
            buffer_size, 
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, 
            &mem_properties
        ); 

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to Map Memory") as *mut u32;

            data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = VulkanApp::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &mem_properties,
        );

        VulkanApp::copy_buffer(
            device,
            submit_queue,
            command_pool,
            staging_buffer,
            index_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_buffer(
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };

        let buffer = unsafe {
            device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Buffer")
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = VulkanApp::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_properties,
            *device_memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
        };

        let buffer_memory = unsafe {
            device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocat buffer memory!")
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind Buffer");
        }

        (buffer, buffer_memory)
    }

    fn copy_buffer(device: &ash::Device, submit_queue: vk::Queue, command_pool: vk::CommandPool, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate Command Buffer")
        };

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin Command Buffer");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];

            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end Command Buffer");
        }

        let submit_info = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            device
                .queue_submit(submit_queue, &submit_info, vk::Fence::null())
                .expect("Failed to Submit Queue.");
            device
                .queue_wait_idle(submit_queue)
                .expect("Failed to wait Queue idle");

            device.free_command_buffers(command_pool, &command_buffers);
        }
    }

    fn find_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            //if (type_filter & (1 << i)) > 0 && (memory_type.property_flags & required_properties) == required_properties {
            //    return i as u32
            // }

            // same implementation
            if (type_filter & (1 << i)) > 0
                && memory_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }

        panic!("Failed to find suitable memory type!")
    }

    fn creaet_uniform_buffers(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, swapchain_image_count: usize) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>();

        let mem_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let mut uniform_buffers = vec![];
        let mut uniform_buffers_memory = vec![];

        for _ in 0..swapchain_image_count {
            let (uniform_buffer, uniform_buffer_memory) = VulkanApp::create_buffer(
                device,
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                &mem_properties,
            );

            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }

        (uniform_buffers, uniform_buffers_memory)
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_bingings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
        }
        ];

        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: ubo_layout_bingings.len() as u32,
            p_bindings: ubo_layout_bingings.as_ptr(),
        };

        unsafe {
            device
                .create_descriptor_set_layout(&ubo_layout_create_info, None)
                .expect("Failed to create Descriptor Set Layout!")
        }
    }
    
    fn create_descriptor_pool(device: &ash::Device, swapchain_images_size: usize) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain_images_size as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: swapchain_images_size as u32,
        }];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO ,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: swapchain_images_size as u32,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        }
    }

    fn update_uniform_buffer(&mut self, current_image: usize, delta_time: f32) {
        // self.uniform_transform.model =
        //     Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Deg(90.0) * delta_time)
        //         * self.uniform_transform.model;

        let ubos = [self.uniform_transform.clone()];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(
                        self.uniform_buffers_memory[current_image],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to Map Memory") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    fn create_descriptor_sets(device: &ash::Device, descriptor_pool: vk::DescriptorPool, descriptor_set_layout: vk::DescriptorSetLayout, uniform_buffers: &Vec<vk::Buffer>, swapchain_images_size: usize, texture_sampler: vk::Sampler, texture_image_view: vk::ImageView) -> Vec<vk::DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: swapchain_images_size as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i],
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as u64,
            }];

            let descriptor_image_info = [vk::DescriptorImageInfo {
                sampler: texture_sampler,
                image_view: texture_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];

            let descriptor_write_sets = [vk::WriteDescriptorSet {
                // transform uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null(),
            },
            vk::WriteDescriptorSet {
                // sampler uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: descriptor_image_info.as_ptr(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
            },
            ];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }

    fn create_texture_image(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, device_memory_properties: &vk::PhysicalDeviceMemoryProperties, image_path: &Path) -> (vk::Image, vk::DeviceMemory, u32) {
        let mut image_object = image::open(image_path).expect("Can not read texture image from file");
        image_object = image_object.flipv();
        let (width, height) = (image_object.width(), image_object.height());
        let image_size =
            (std::mem::size_of::<u8>() as u32 * width * height * 4) as vk::DeviceSize;

        let mip_levels = ((::std::cmp::max(width, height) as f32)
            .log2()
            .floor() as u32)
            + 1;

        let image_data = match &image_object {
            image::DynamicImage::ImageLuma8(_)
            | image::DynamicImage::ImageBgr8(_)
            | image::DynamicImage::ImageRgb8(_) => image_object.to_rgba8().into_raw(),
            image::DynamicImage::ImageLumaA8(_)
            | image::DynamicImage::ImageBgra8(_)
            | image::DynamicImage::ImageRgba8(_) => image_object.to_bytes(),
            _ => panic!("Unsupported image format")
        };

        if image_size <= 0 {
            panic!("Failed to load texture image!")
        }

        let (staging_buffer, staging_buffer_memory) = VulkanApp::create_buffer(
            device,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device_memory_properties,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut u8;

            data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (texture_image, texture_image_memory) = VulkanApp::create_image(
            device,
            width,
            height,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );

        VulkanApp::transition_image_layout(device, command_pool, submit_queue, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mip_levels);
        VulkanApp::copy_buffer_to_image(device, command_pool, submit_queue, staging_buffer, texture_image, width, height);
        // VulkanApp::transition_image_layout(device, command_pool, submit_queue, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        
        VulkanApp::generate_mipmaps(device, command_pool, submit_queue, texture_image, width, height, mip_levels);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (texture_image, texture_image_memory, mip_levels)
    }

    fn create_image(
        device: &ash::Device, 
        width: u32, 
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling, 
        usage: vk::ImageUsageFlags, 
        required_memory_properties: vk::MemoryPropertyFlags, 
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties) -> (vk::Image, vk::DeviceMemory) {
            let image_create_info = vk::ImageCreateInfo {
                s_type: vk::StructureType::IMAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageCreateFlags::empty(),
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D { width: width, height: height, depth: 1 },
                mip_levels: mip_levels,
                array_layers: 1,
                samples: num_samples,
                tiling,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                initial_layout: vk::ImageLayout::UNDEFINED,
            };

            let texture_image = unsafe {
                device
                    .create_image(&image_create_info, None)
                    .expect("Failed to create Texture Image!")
            };

            let image_memory_requirement =
            unsafe { device.get_image_memory_requirements(texture_image) };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: image_memory_requirement.size,
            memory_type_index: VulkanApp::find_memory_type(
                image_memory_requirement.memory_type_bits,
                required_memory_properties,
                *device_memory_properties,
            ),
        };

        let texture_image_memory = unsafe {
            device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate Texture Image memory!")
        };

        unsafe {
            device
                .bind_image_memory(texture_image, texture_image_memory, 0)
                .expect("Failed to bind Image Memmory!");
        }

        (texture_image, texture_image_memory)
    }

    fn begin_single_time_command(device: &ash::Device, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY ,
            command_buffer_count: 1,
        };

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),   
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        command_buffer
    }

    fn end_single_time_command(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, command_buffer: vk::CommandBuffer) {
        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let buffers_to_submit = [command_buffer];
        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            device
                .queue_submit(submit_queue, &submit_infos, vk::Fence::null())
                .expect("Failed to Queue Submit!");
            device
                .queue_wait_idle(submit_queue)
                .expect("Failed to wait Queue idle!");
            device.free_command_buffers(command_pool, &buffers_to_submit);
        }
    }

    fn transition_image_layout(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, image: vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32) {
       let command_buffer = VulkanApp::begin_single_time_command(device, command_pool);

       let src_access_mask;
       let dst_access_mask;
       let source_stage;
       let destination_stage;

       if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
       } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
       && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
           destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
       } else {
           panic!("Unsupported layout transition!")
       }

       let image_barriers = [vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask,
        dst_access_mask,
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        },
    }];

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );
    }

    VulkanApp::end_single_time_command(device, command_pool, submit_queue, command_buffer);
    }

    fn copy_buffer_to_image(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
        let command_buffer = VulkanApp::begin_single_time_command(device, command_pool);

        let buffer_image_regions = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask:  vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }];

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_regions,
            );
        }

        VulkanApp::end_single_time_command(device, command_pool, submit_queue, command_buffer);
    }

    fn create_texture_image_view(device: &ash::Device, texture_image: vk::Image, mip_levels: u32) -> vk::ImageView {
        let texture_image_view =
            VulkanApp::create_image_view(device, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR, mip_levels);
        texture_image_view
    }


    fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags, mip_levels: u32,) -> vk::ImageView {
        let imageview_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
        };

        unsafe {
            device
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        }
    }

    fn create_depth_resources(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, command_pool: vk::CommandPool, submit_queue: vk::Queue, swapchain_extent: vk::Extent2D, device_memory_properties: &vk::PhysicalDeviceMemoryProperties
    ) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
        let depth_format = VulkanApp::find_depth_format(instance, physical_device);
        let (depth_image, depth_image_memory) = VulkanApp::create_image(
            device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );

        let depth_image_view = VulkanApp::create_image_view(
            device,
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        );

        (depth_image, depth_image_view, depth_image_memory)
    }

    fn find_depth_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
        VulkanApp::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn find_supported_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice, candidate_formats: &[vk::Format], tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties = unsafe { instance.get_physical_device_format_properties(physical_device, format) };

            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features) {
                    return format.clone();
            } else if tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features) {
                    return format.clone();
            }
        }

        panic!("Failed to find supported format!")
    }

    fn generate_mipmaps(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, image: vk::Image, tex_width: u32, tex_height: u32, mip_levels: u32) {
        let command_buffer = VulkanApp::begin_single_time_command(device, command_pool);

        let mut image_barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
            .build();
        
        let mut mip_width = tex_width as i32;
        let mut mip_height = tex_height as i32;
        
        for i in 1..mip_levels {
            image_barrier.subresource_range.base_mip_level = i - 1;
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            let blits = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: i - 1, base_array_layer: 0, layer_count: 1 },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { 
                        x: mip_width,
                        y: mip_height,
                        z: 1
                    }
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: i32::max(mip_width / 2, 1),
                        y: i32::max(mip_height / 2, 1),
                        z: 1,
                    },
                ],
            }];

            unsafe {
                device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );
            }

            image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            mip_width = i32::max(mip_width / 2, 1);
            mip_height = i32::max(mip_height / 2, 1);
        }

        image_barrier.subresource_range.base_mip_level = mip_levels - 1;
        image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier.clone()],
            );
        }

        VulkanApp::end_single_time_command(device, command_pool, submit_queue, command_buffer);
    }

    fn check_mipmap_support(instance: &ash::Instance, physcial_device: vk::PhysicalDevice, image_format: vk::Format) {
        let format_properties = unsafe {
            instance.get_physical_device_format_properties(physcial_device, image_format)
        };

        let is_sample_image_filter_linear_support = format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

        if is_sample_image_filter_linear_support == false {
            panic!("Texture Image format does not support linear blitting!")
        }
    }

    fn create_texture_sampler(device: &ash::Device, mip_levels: u32) -> vk::Sampler {
        let sampler_create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            mip_lod_bias: 0.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
        };

        unsafe {
            device
                .create_sampler(&sampler_create_info, None)
                .expect("Failed to create Sampler!")
        }
    }

    #[allow(dead_code)]
    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn draw_frame(&mut self, delta_time:  f32) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        let (image_index, _is_suboptimal) = unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for fence");

            let result: VkResult<(u32, bool)>  = self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
            std::u64::MAX, 
            self.image_available_semaphores[self.current_frame], 
                    vk::Fence::null()
            );

            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                }
            }
        };

        self.update_uniform_buffer(image_index as usize, delta_time);
        
        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");
            
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
            )
            .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        let result: VkResult<bool> = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };

        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
   
    fn main_loop(mut self, event_loop: EventLoop<()>) {
        let mut tick_counter = utils::fps_limiter::FPSLimiter::new();

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
                    self.window.request_redraw();
                },
                Event::RedrawRequested(_window_id) => {
                    tick_counter.tick_frame();
                    let delta_time = tick_counter.delta_time();
                    self.draw_frame(delta_time);
                },
                _ => {}
            }
        })
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.cleanup_swapchain();

            // self.device
            //     .destroy_descriptor_pool(self.descriptor_pool, None);

            for i in 0..self.uniform_buffers.len() {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_sampler(self.texture_sampler, None);
            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
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

    let mut vulkan_app = VulkanApp::new(window); 
    vulkan_app.main_loop(event_loop);
}
