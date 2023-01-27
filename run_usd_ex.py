import os
import time
import warp as wp
import numpy as np
from pxr import Usd
import usd_utils
from warpforce import WarpCrowd

# wp.config.mode = "debug"
# wp.config.print_launches = True
# wp.config.verify_cuda = True

#### Initialize WARP #####
wp.init()

def run_class():

    # USD Scene stuff
    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "simple_env.usd"))
    points, faces = usd_utils.get_mesh(usd_stage, objs = ['/World/Plane'])
    stage = os.path.join(os.path.dirname(__file__), "warpcrowd_output.usd")
    renderer = wp.render.UsdRenderer(stage, upaxis='z')

    wc = WarpCrowd()
    wc.demo_agents(m=30, n=30)

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([-30.8,0.01582,0.0])

    radius = wc.radius

    start = time.time()
    sim_time = 0
    for i in range(750):
        wc.compute_step()
        sim_time = render(False, renderer, wc.mesh, wc.xnew_wp, radius, sim_time, wc.dt)
    print(time.time()-start)
    renderer.save()

def render(is_live, renderer, mesh, positions, sim_margin, sim_time, dt):
    
    with wp.ScopedTimer("render", detailed=False):
        time = 0.0 if is_live else sim_time 
        
        renderer.begin_frame(time)
        renderer.render_mesh(name="mesh", points=mesh.points.numpy(), indices=mesh.indices.numpy())
        renderer.render_points(name="points", points=positions.numpy(), radius=sim_margin)
        renderer.end_frame()

    sim_time += dt
    return sim_time

run_class()
