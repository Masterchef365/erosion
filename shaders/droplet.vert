#version 450
#extension GL_EXT_multiview : require

layout(binding = 0) uniform SceneData {
    mat4 camera[2];
    float anim;
};

layout(location = 0) out vec3 fragColor;

void main() {
    //gl_Position = camera[gl_ViewIndex] * model * vec4(inPosition, 1.0);
    gl_Position = camera[gl_ViewIndex] * vec4(inPosition, 1.0);
    fragColor = inColor;
}

