Steps:
1. Initialize the map with one or more of:
    * Perlin noise
    * Randomly placed radial bumps
2. Read from heightmap, Write to sediment diff
3. Blur sediment diff
4. Read from sediment diff, write to heightmap

Maybe for the first impl, we'll just:
1. Initialize with random radial bumps and perlin noise
2. Read from the heightmap, write to the half-resolution sediment diff
3. Sample from sediment diff and write to heightmap
4. Repeat from step 2 N times


In even more detail:
1. Initialize heightmap with perlin noise and a set of SDFs corresponding to land forms
2. Initialize particles
3. Step particles, reading from the heightmap arbitrarily, and writing to the erosion map.
4. Blur the erosion map, with different radii according to the sign
5. Add the erosionmap to the height map
6. Goto 3
7. Optionally do the display code too... So that you sample that heightmap texture from a dense grid of triangles... Maybe do vertex pulling as well... Yeahh... Wait what about indices,, uhhh hmm

Notes:
* It's really the deposition-erosion map
* Try without the blur image and just use a backbuffer, i32 heightmap, and atomicAdd - that way you can also directly render from the heighmap front-buffer on the opposite frame...

# Implementation steps
1. Visualization testing
    * Download a heightmap from the internet and upload it to the heightmap buffer
    * Select a few random particle positions
    
```rust
struct InitSettings {
    /// Random seed
    pub seed: u64,
    /// Noise scale
    pub noise_scale: f32,
    /// Noise vertical amplitude
    pub noise_amplitude: f32,
    /// Number of hills (randomly placed)
    pub n_hills: u32,
    /// Falloff rate for the curve
    pub hill_falloff: f32,
}

// TODO: Builder pattern?
struct SimulationSettings {
    /// Inertia
    pub inertia: u32,
    /// Minimum slope for capacity calculation
    pub min_slope: u32,
    /// Capacity for droplets to carry material
    pub capacity_const: u32,
    /// Sediment dropped beyond capacity
    pub deposition: u32,
    /// Sediment picked up under capacity
    pub erosion: u32,
    /// Force of gravity
    pub gravity: u32,
    /// Evaporation rate
    pub evaporation: u32, // TODO: 1- evap for speed!
}

impl ErosionSim {
    pub fn new(core: SharedCore, width: u32, height: u32, settings: &InitSettings) {}

    pub fn step(&mut self, cmd: vk::CommandBuffer, settings: &SimulationSettings, iters: u32) {}

    pub fn reset(&mut self, cmd: vk::CommandBuffer, settings: &InitSettings) {}

    pub fn particle_buffer(&mut self) -> vk::Buffer {}
    pub fn heightmap_buffer(&mut self) -> vk::Buffer {}

    pub fn download_particles(&mut self) -> Vec<Particle> {}
    pub fn download_heightmap(&mut self) -> Vec<f32> {}

    //pub fn upload_heightmap(&mut self, data: &[f32]);
    //pub fn upload_particles(&mut self, particles: &[Particle]);
}
```
