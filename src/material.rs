use ash::vk;

struct Material {
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout
}