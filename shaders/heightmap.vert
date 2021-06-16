
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : require

layout(binding = 0) uniform Animation {
    mat4 camera[2];
    float anim;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(binding = 2) uniform sampler2D tex;

void main() {
    vec3 pos = inPosition;
    pos.y += float(texture(tex, inColor.xy).r) * 30.;
    gl_Position = camera[gl_ViewIndex] * vec4(pos, 1.0);
    fragColor = inColor;
}

