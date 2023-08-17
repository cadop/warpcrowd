import os
import time
import warp as wp
import numpy as np
from pxr import Usd
import usd_utils
from warpforce import WarpCrowd
from warp.render import render_usd

#### Initialize WARP #####
wp.init()

def run_class():

    # USD Scene stuff
    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "Collected_simple_env2/simple_env2.usd"))
    
    points, faces = usd_utils.get_all_stage_mesh(usd_stage)
    stage = os.path.join(os.path.dirname(__file__), "warpcrowd_output.usd")
    renderer = render_usd.UsdRenderer(stage, up_axis='z')

    wc = WarpCrowd(up_axis='z')
    wc.demo_agents(m=3, n=3)

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([[-30.8,0.01582,0.0]])

    radius = wc.radius

    start = time.time()
    sim_time = 0

    # Render an initial frame to insert static mesh
    renderer.begin_frame(0.0)
    renderer.render_mesh(name="mesh", points=wc.mesh.points.numpy(), indices=wc.mesh.indices.numpy())
    renderer.end_frame()
    
    for i in range(1):
        wc.compute_step()
        sim_time = render(False, renderer, wc.mesh, wc.xnew_wp, radius, sim_time, wc.dt)
    
    print(time.time()-start)
    renderer.save()

def render(is_live, renderer, mesh, positions, sim_margin, sim_time, dt):
    
    with wp.ScopedTimer("render", detailed=False):
        time = 0.0 if is_live else sim_time 
        
        renderer.begin_frame(time)
        renderer.render_points(name="points", points=positions.numpy(), radius=sim_margin)
        renderer.end_frame()

    sim_time += dt
    return sim_time

run_class()