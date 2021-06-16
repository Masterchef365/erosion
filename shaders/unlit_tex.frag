#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;
layout(binding = 2) uniform sampler2D tex;

void main() {
    outColor = vec4(vec3(texture(tex, fragColor.xy).r), 1.0);
}
