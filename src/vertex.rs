use ash::vk;
use memoffset::offset_of;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    tex_coord: [f32; 2]
}

impl Vertex {
    pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        }]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, pos) as u32
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, tex_coord) as u32,
            },
        ]
    }
}

pub const VERTICES_DATA: [Vertex; 8] = [
    Vertex {
        pos: [-0.75, -0.75, 0.0],
        color: [1.0, 0.0, 0.0],
        tex_coord: [1.0, 0.0],
    },
    Vertex {
        pos: [0.75, -0.75, 0.0],
        color: [0.0, 1.0, 0.0],
        tex_coord: [0.0, 0.0],
    },
    Vertex {
        pos: [0.75, 0.75, 0.0],
        color: [0.0, 0.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
    Vertex {
        pos: [-0.75, 0.75, 0.0],
        color: [1.0, 1.0, 1.0],
        tex_coord: [1.0, 1.0],
    },

    Vertex {
        pos: [-0.75, -0.75, -0.9],
        color: [1.0, 0.0, 0.0],
        tex_coord: [0.0, 0.0],
    },
    Vertex {
        pos: [0.75, -0.75, -0.9],
        color: [0.0, 1.0, 0.0],
        tex_coord: [1.0, 0.0],
    },
    Vertex {
        pos: [0.75, 0.75, -0.9],
        color: [0.0, 0.0, 1.0],
        tex_coord: [1.0, 1.0],
    },
    Vertex {
        pos: [-0.75, 0.75, -0.9],
        color: [1.0, 1.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
];

pub const INDICES_DATA: [u32; 12] = [
    4, 5, 6, 6, 7, 4,
    0, 1, 2, 2, 3, 0
];