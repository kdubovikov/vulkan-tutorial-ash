use std::path::Path;

use ash::vk;

use crate::vertex::Vertex;

pub mod platform;
pub mod debug;
pub mod tools;
pub mod shader;
pub mod fps_limiter;
pub mod command;
pub mod buffer;

pub fn load_model(path: &Path) -> (Vec<Vertex>, Vec<u32>) {
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

pub fn find_memory_type(
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