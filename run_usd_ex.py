import os
import time
import warp as wp
import numpy as np
from pxr import Usd
import usd_utils
from warpforce import WarpCrowd

from warp_utils import NewRenderer
#### Initialize WARP #####
wp.init()

def run_class():

    up_axis = 'y'
    stage = os.path.join(os.path.dirname(__file__), "warpcrowd_output.usd")
    renderer = NewRenderer(stage, up_axis=up_axis)

    # USD Scene stuff
    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "examples/simple_env.usda"))
    points, faces = usd_utils.get_all_stage_mesh(usd_stage)


    wc = WarpCrowd(up_axis=up_axis)
    wc.demo_agents(m=100, n=100, s=1.6)

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([[-80.8,-50.0,0.0]])

    radi = wc.agents_radi.flatten()

    start = time.time()
    sim_time = 0
    num_sim_steps = 1200

    # # Render an initial frame to insert static mesh
    renderer.begin_frame(0.0)
    renderer.render_mesh(name="mesh", points=wc.mesh.points.numpy(), indices=wc.mesh.indices.numpy())
    renderer.end_frame()
    
    for i in range(num_sim_steps):
        wc.compute_step()
        heading = wc.agents_hdir_wp.numpy()
        sim_time = render(False, renderer, wc.mesh, wc.xnew_wp, heading, radi, sim_time, wc.dt)
    
    print(f'Computation time of {num_sim_steps} for {len(wc.agents_pos)} agents took: {time.time()-start} seconds, for {((time.time()-start)/num_sim_steps)*1000}ms per step')
    renderer.save()

def render(is_live, renderer, mesh, positions, headings, sim_margin, sim_time, dt):
    
    with wp.ScopedTimer("render", detailed=False):
        time = 0.0 if is_live else sim_time 
        
        renderer.begin_frame(time)

        # Render capsules
        renderer.render_capsules(name="CapsuleAgent", 
                                 points=positions.numpy(), 
                                 radius=sim_margin, 
                                 half_height=.45, 
                                 orientations=headings)
        
        # Render points, a bit faster than capsules
        # renderer.render_points(name="points", points=positions.numpy(), radius=sim_margin)
        renderer.end_frame()

    sim_time += dt
    return sim_time


run_class()