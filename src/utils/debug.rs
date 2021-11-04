use std::{ffi::{CStr, c_void}, ptr};

use ash::{extensions::ext::DebugUtils, vk};

pub struct ValidationInfo {
    pub enabled: bool,
    pub required_validation_layers: [&'static str; 1]
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[VERBOSE]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[WARNING]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[ERROR]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[INFO]",
        _ => "[UNKNOWN]"
    };

    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
        _ => "[UNKNOWN]"
    };

    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("DEBUG {} {} {:?}", severity, types, message);

    vk::FALSE
}

pub fn check_validation_layer_support(
    entry: &ash::Entry,
    required_validation_layers: &Vec<&str>,
) -> bool {
    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Failed to enumerate instance layers properties");

    if layer_properties.len() <= 0 {
        eprintln!("No available layers");
        return false;
    }

    for required_layer_name in required_validation_layers.iter() {
        let mut is_layer_found = false;

        for layer_property in layer_properties.iter() {
            let test_layer_name = super::tools::vk_to_string(&layer_property.layer_name);
            if (*required_layer_name) == test_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if is_layer_found == false {
            return false;
        }
    }

    true
}

pub fn setup_debug_utils(
    is_enable_debug: bool,
    entry: &ash::Entry,
    instance: &ash::Instance
) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = DebugUtils::new(entry, instance);

    if is_enable_debug {
        let messenger_ci = populate_debug_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Failed to create debug utils messenger")
        };

        (debug_utils_loader, utils_messenger)
    } else {
        (debug_utils_loader, vk::DebugUtilsMessengerEXT::null())
    }
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    }
}