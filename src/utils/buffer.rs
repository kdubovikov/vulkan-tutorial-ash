use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    allocator: Arc<Mutex<Allocator>>,
    device: Arc<ash::Device>,
    allocation_name: String 
}

impl AllocatedBuffer {
    pub fn create(
        allocator: Arc<Mutex<Allocator>>,
        allocation_name: &str,
        device: Arc<ash::Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation
    ) -> AllocatedBuffer {
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

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    
        let allocation_create_info = AllocationCreateDesc {
            name: allocation_name,
            requirements,
            location: memory_location,
            linear: true,
        };
        
        let allocation = allocator.lock().unwrap().allocate(&allocation_create_info).unwrap();
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap() };

        AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
            allocator,
            device,
            allocation_name: allocation_name.to_string()
        }
    }

    pub fn create_from_data<T>(
        allocator: Arc<Mutex<Allocator>>,
        allocation_name: &str, 
        instance: &ash::Instance, 
        device: Arc<ash::Device>, 
        physical_device: vk::PhysicalDevice, 
        command_pool: vk::CommandPool, 
        submit_queue: vk::Queue, 
        usage: vk::BufferUsageFlags,
        data: &[T]
    ) -> AllocatedBuffer where T: Copy {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let mem_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let staging_buffer = AllocatedBuffer::create(
            allocator.clone(),
            &(allocation_name.clone().to_owned() + " staging buffer"),
            device.clone(), 
            buffer_size, 
            vk::BufferUsageFlags::TRANSFER_SRC, 
            MemoryLocation::CpuToGpu,
        );

        staging_buffer.copy_data_from(&data, data.len());

        let buffer = AllocatedBuffer::create(
            allocator.clone(),
            allocation_name,
            device.clone(),
            buffer_size,
            usage,
            MemoryLocation::GpuOnly, 
        );

        staging_buffer.copy_to(
            device,
            submit_queue,
            command_pool,
            &buffer,
            buffer_size,
        );

        buffer
    }

    pub fn copy_to(&self, device: Arc<ash::Device>, submit_queue: vk::Queue, command_pool: vk::CommandPool, dst_buffer: &AllocatedBuffer, size: vk::DeviceSize) {
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
    
            device.cmd_copy_buffer(command_buffer, self.buffer, dst_buffer.buffer, &copy_regions);
    
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
   
    pub fn copy_data_from<T>(&self, src: &[T], count: usize) where T: Copy {
        if let Some(allocation) = &self.allocation {
            unsafe {
                let size = (src.len() * std::mem::size_of::<T>()) as _;
                let data_ptr = allocation.mapped_ptr().unwrap().cast().as_ptr();
                let mut align = ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as _, size);
                align.copy_from_slice(src);
                // std::ptr::copy_nonoverlapping(
                //     src,
                //     data_ptr,
                //     count,
                // );
            }
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        self.allocator.lock().unwrap().free(self.allocation.take().unwrap()).unwrap();
        unsafe { self.device.destroy_buffer(self.buffer, None) };
    }
}
