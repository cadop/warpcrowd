import time
import warp as wp
import numpy as np

import mesh_utils
from warpforce import WarpCrowd

#### Initialize WARP #####
wp.init()

def run_class():

    points, faces = mesh_utils.load_obj('examples/simple_env.obj')

    wc = WarpCrowd()
    wc.demo_agents(m=100, n=100, s=1.6)

    wc.nagents = len(wc.agents_pos)
    wc.configure_params() # Call to setup params for warp, use after defining agents

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([[-80.8,-50.0,0.0]])

    start = time.time()
    num_sim_steps = 800

    # Run simulation Steps
    for i in range(num_sim_steps):
        wc.compute_step()
        # Get Agent Positions
        pos = wc.xnew_wp

    print(f'Computation time of {num_sim_steps} for {len(wc.agents_pos)} agents took: {time.time()-start} seconds, for {((time.time()-start)/num_sim_steps)*1000}ms per step')


run_class()
