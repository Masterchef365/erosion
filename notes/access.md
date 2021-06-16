**For heightmap and particle visualization shaders**:
```glsl
layout (binding = 0) uniform SceneData {
    mat4 camera[2];
    float anim;
};

layout (binding = 1) buffer Droplets {
    Droplet droplets[];
};

layout (binding = 2) uniform sampler2D heightmap;
```

**For simulation kernels**:
```glsl
layout (binding = 0) uniform Settings {
    // Varies by kernel, Reset/Init get different data than SimStep settings structures
};

layout (binding = 1) buffer Droplets {
    Droplet droplets[];
};

layout (binding = 2) uniform sampler2D heightmap_sm;
layout (binding = 3, r32f) uniform image2D heightmap;
layout (binding = 4, r32f) uniform image2D erosionmap;
```
