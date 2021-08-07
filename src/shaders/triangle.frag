#version 460


layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 in_vt;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(tex_sampler, in_vt);
}