use std::{path::Path, ptr};

use ash::vk;
use image::GenericImageView;

use crate::utils;


pub fn create_texture_image(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, device_memory_properties: &vk::PhysicalDeviceMemoryProperties, image_path: &Path) -> (vk::Image, vk::DeviceMemory, u32) {
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

    let (staging_buffer, staging_buffer_memory) = utils::buffer::create_buffer(
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

    let (texture_image, texture_image_memory) = create_image(
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

    transition_image_layout(device, command_pool, submit_queue, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mip_levels);
    copy_buffer_to_image(device, command_pool, submit_queue, staging_buffer, texture_image, width, height);
    // VulkanApp::transition_image_layout(device, command_pool, submit_queue, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    
    generate_mipmaps(device, command_pool, submit_queue, texture_image, width, height, mip_levels);

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    (texture_image, texture_image_memory, mip_levels)
}

pub fn create_image(
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
        memory_type_index: utils::find_memory_type(
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

pub fn create_texture_image_view(device: &ash::Device, texture_image: vk::Image, mip_levels: u32) -> vk::ImageView {
    let texture_image_view =
        create_image_view(device, texture_image, vk::Format::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR, mip_levels);
    texture_image_view
}

pub fn create_image_views(device: &ash::Device, surface_format: vk::Format, images: &Vec<vk::Image>) -> Vec<vk::ImageView> {
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

pub fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags, mip_levels: u32,) -> vk::ImageView {
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

pub fn get_max_usable_sample_count(instance: &ash::Instance, physcial_device: vk::PhysicalDevice) -> vk::SampleCountFlags {
    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physcial_device) };

    let count = std::cmp::min(
        physical_device_properties
            .limits
            .framebuffer_color_sample_counts,
        physical_device_properties
            .limits
            .framebuffer_depth_sample_counts,
    );

    if count.contains(vk::SampleCountFlags::TYPE_64) {
        return vk::SampleCountFlags::TYPE_64;
    }
    if count.contains(vk::SampleCountFlags::TYPE_32) {
        return vk::SampleCountFlags::TYPE_32;
    }
    if count.contains(vk::SampleCountFlags::TYPE_16) {
        return vk::SampleCountFlags::TYPE_16;
    }
    if count.contains(vk::SampleCountFlags::TYPE_8) {
        return vk::SampleCountFlags::TYPE_8;
    }
    if count.contains(vk::SampleCountFlags::TYPE_4) {
        return vk::SampleCountFlags::TYPE_4;
    }
    if count.contains(vk::SampleCountFlags::TYPE_2) {
        return vk::SampleCountFlags::TYPE_2;
    }

    vk::SampleCountFlags::TYPE_1
}

pub fn generate_mipmaps(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, image: vk::Image, tex_width: u32, tex_height: u32, mip_levels: u32) {
    let command_buffer = utils::command::begin_single_time_command(device, command_pool);

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

    utils::command::end_single_time_command(device, command_pool, submit_queue, command_buffer);
}

pub fn check_mipmap_support(instance: &ash::Instance, physcial_device: vk::PhysicalDevice, image_format: vk::Format) {
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

pub fn create_texture_sampler(device: &ash::Device, mip_levels: u32) -> vk::Sampler {
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


pub fn transition_image_layout(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, image: vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32) {
        let command_buffer = utils::command::begin_single_time_command(device, command_pool);

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

    utils::command::end_single_time_command(device, command_pool, submit_queue, command_buffer);
 }

fn copy_buffer_to_image(device: &ash::Device, command_pool: vk::CommandPool, submit_queue: vk::Queue, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
    let command_buffer = utils::command::begin_single_time_command(device, command_pool);

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

    utils::command::end_single_time_command(device, command_pool, submit_queue, command_buffer);
}