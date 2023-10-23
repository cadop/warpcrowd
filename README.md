# warpcrowd
`warpcrowd` is a python script for running crowd simulations on the GPU. It is relatively simple (on purpose) with the simulation calculations (social forces) done in one file. Additional files provide some utilities and examples of how to run the simulation. 

The script is purely python and relies on NVIDIA Warp for GPU acceleration. However, the USD run example shows how to load and render an environment with the simulation saved to a USD file that can be loaded in Omniverse. 

## Use
There are example files in the `examples` folder. The two sample scripts are `run_mesh_ex.py` and `run_usd_ex.py`. These both by default reference a sample file, although you can change to most other files. Meshes will often be tricky, and although there is a helper function for converting ngons to triangles, its best at the beginning stages to just use triangles. 

## Features
The social forces implementation is in 3D, but currently resets the vertical force to 0 to keep agents on the ground. The array of agent goals can be set by the user at their desired timestep (e.g. using A*), but in the current samples it is a static goal. The wall forces are calculated by all triangles in the scene (within a distance threshold), which is obviously not the most efficient way. 

The main point of these two statements is to put into context the performance of the simulator. With environments of over 20k triangles and 10,000 agents, calculation and saving capsules for 600 timesteps takes around 7 seconds. In some light experiments, it seems the calculation time is nearly the same as the number of agents increase as long as the number of CUDA cores (and memory) are greater than the number of agents. 

# Requirements
- `warp-lang`
- `numpy`
- `usd-core` (for USD environments)
- `pywavefront` (only needed if dealing with meshes like .obj)
