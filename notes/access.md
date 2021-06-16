```glsl
layout (binding = 0) uniform SceneData {
    mat4 camera[2];
    float anim;
};

layout (binding = 1) buffer Droplets {
    Droplet droplets[];
};

layout (binding = 2, r32f) uniform readonly image2D heightmap;
layout (binding = 3, r32f) uniform image2D erosionmap;
layout (binding = 4) uniform sampler2D heightmap_sm;
```


Droplet vert:
* Droplets
* SceneData

HeightMap vert:
* SceneData
* Heightmap sampler

HeightMap frag:
* Heightmap

DropletInit:
* Droplets

HeightMapInit:
* HeightMap image

SimStep:
* Droplets
* HeightMap image

... It's gotta have an area of effect ... 
So maybe we _do_ have an erosionmap, but each pixel is just 


Sim step is calculating the next position, and also how much material is to be added or removed from the position the particles were _before_ the step.
