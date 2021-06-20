# GPGPU terrain generator
This project attempts to implement _Implementation of a method for hydraulic
erosion_ by Hans Theobald Beyer, available in [PDF form](https://www.firespark.de/resources/downloads/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf)

Keep in mind that this is just a hobby project and may never be of any production quality!

# To-do
* Literally any kind of export function
    * PNG with 8- or 16-bit single-channel should work alright
    * That and just a "u32 width, then float image data" output
* Figure out how to keep the wells from happening
* Make certain there's no race condition on `imageStore(erosion, ...)` for the `sim_step` kernel
* Correct falloff function for the erosion blur (so that no material is lost)
* Kernel size settings
* Non-square simulations
* Optimizations out the wazoo
