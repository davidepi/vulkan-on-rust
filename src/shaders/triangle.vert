#version 460

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 vertex_pos;
layout(location = 1) in vec3 vertex_color;
layout(location = 2) in vec2 in_vt;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 out_vt;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(vertex_pos, 1.0);
    frag_color = vertex_color;
    out_vt = in_vt;
}