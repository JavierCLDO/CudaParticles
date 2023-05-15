# CudaParticles

https://github.com/FenomenCoyote/FenomenEngine/assets/47754234/25bbd43f-037e-44b0-823b-7c43b65fed3c

This was made as an exercise to learn CUDA C++. 

Using only SDL2 to create a window and render a pixel array (rendering on a 1024\*1024 window), I run a simulation with more than half a million particles using the GPU to compute the collisions, the explosions and the rendering, but without using a shader (examples of shaders acomplishing particle collisions can be found at shadertoy).

For the video demo I used a 3080, and I streched the simulation and performance to the best of my knowldge of CUDA (which is not much at the moment). The simulation runs at 30 fps and uses 4 simulation substeps. 
