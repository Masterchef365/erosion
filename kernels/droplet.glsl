float rand(vec2 c){
    return fract(sin(dot(c.xy, vec2(19.9898,71.233))) * 4758.5453);
}

struct Droplet {
    // Position
    vec2 pos;
    // Direction
    vec2 dir;
    // Velocity
    float vel;
    // Water
    float water;
    // Sediment
    float sediment;
};

Droplet init_droplet(vec2 seed) {
    Droplet droplet;
    droplet.pos = vec2(
        rand(seed * vec2(23.123, 34.1241)),
        rand(seed * vec2(95.123, 91.321))
    );
    droplet.dir = vec2(
        rand(seed * vec2(13.13, 14.141)),
        rand(seed * vec2(25.12, 21.31))
    ) * 2. - 1.;
    droplet.vel = 1.;
    droplet.water = 1.;
    droplet.sediment = 0.;
    return droplet;
}
