#version 450
#extension GL_EXT_multiview : require

#include "../kernels/droplet.glsl"

layout (binding = 0) uniform SceneData {
    mat4 camera[2];
    float anim;
};

layout (binding = 1) buffer Droplets {
    Droplet droplets[];
};

layout (binding = 2) uniform sampler2D heightmap;

layout(location = 0) out vec3 fragColor;

void main() {
    Droplet droplet = droplets[gl_VertexIndex];

    float height = texture(heightmap, droplet.pos).r; 

    // TODO: Standardize this magic!
    height = height * 30. + 0.5;

    vec3 pos = vec3(droplet.pos.x, height, droplet.pos.y);

    gl_Position = camera[gl_ViewIndex] * vec4(pos, 1.0);

    fragColor = vec3(0., 1., 0.);
}

