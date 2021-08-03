#version 460

layout(location = 0) in vec3 vertex_pos;
layout(location = 1) in vec3 vertex_color;
layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(vertex_pos, 1.0);
    frag_color = vertex_color;
}