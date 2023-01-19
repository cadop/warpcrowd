import warp as wp
import numpy as np
import time
import forces as sf
import os
from pxr import Usd, UsdGeom, Sdf, Gf, UsdLux

import usd_utils

# wp.config.mode = "debug"
# wp.config.print_launches = True
# wp.config.verify_cuda = True

#### Initialize WARP #####
wp.init()
device = "cuda"
# device = "cpu"

def run():

    # generate n number of agents
    nagents = 9
    # set radius
    radius = 1
    # set mass
    mass = 80
    # set pereption radius
    perception_radius = 6

    # Set agent destination
    goal = [-30.8,0.01582,0.0]

    s = radius*2
    pos = [ [0,0,0],
            [0,1*s,0],
            [0,2*s,0],
            [1*s,0,0],
            [1*s,1*s,0],
            [1*s,2*s,0],
            [2*s,0,0],
            [2*s,2*s,0],
            [2*s,1*s,0],
            ]

    # randomly set position
    agents_pos = np.asarray([np.array([x,x,0]) for x in range(nagents)])
    agents_pos = np.asarray(pos, dtype=np.float64)

    agents_vel = np.asarray([np.array([0,0,0]) for x in range(nagents)])
    agents_radi = np.asarray([radius for x in range(nagents)])
    agents_mass = [mass for x in range(nagents)]
    agents_percept = np.asarray([perception_radius for x in range(nagents)])
    agents_goal = np.asarray([np.array(goal, dtype=float) for x in range(nagents)])

    dt = 1.0/30.0

    agent_force = wp.zeros(shape=nagents,device=device, dtype=wp.vec3)

    agents_pos = wp.array(agents_pos, device=device, dtype=wp.vec3)
    agents_vel = wp.array(agents_vel, device=device, dtype=wp.vec3)
    agents_goal = wp.array(agents_goal, device=device, dtype=wp.vec3)
    agents_radi = wp.array(agents_radi, device=device, dtype=float)
    agents_mass = wp.array(agents_mass, device=device, dtype=float)

    xnew = wp.zeros_like(agents_pos)
    vnew = wp.zeros_like(agents_pos)

    # USD Scene stuff
    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "simple_env.usd"))
    points, faces = usd_utils.get_mesh(usd_stage)
    stage = os.path.join(os.path.dirname(__file__), "warp_sf_output.usd")
    renderer = wp.render.UsdRenderer(stage, upaxis='z')

    grid = wp.HashGrid(dim_x=nagents, dim_y=nagents, dim_z=1, device=device)

    points = np.asarray(points)
    faces = np.asarray(faces)

    # Init mesh for environment collision
    mesh = wp.Mesh( points=wp.array(points, dtype=wp.vec3, device=device),
                    indices=wp.array(faces, dtype=int ,device=device)
                  )

    start = time.time()

    sim_time = 0
    for i in range(750):
            
        grid.build(points=agents_pos, radius=radius)

        # launch kernel
        wp.launch(kernel=sf.get_forces,
                dim=nagents,
                inputs=[agents_pos, agents_vel, agents_goal, agents_radi, 
                        agents_mass, dt, perception_radius, grid.id, mesh.id],
                outputs=[agent_force],
                device=device
                )
        # Given the forces, integrate for pos and vel
        wp.launch(kernel=sf.integrate,
                dim=nagents,
                inputs=[agents_pos, agents_vel, agent_force, dt],
                outputs=[xnew, vnew],
                device=device
                )
    
        agents_pos = xnew
        agents_vel = vnew

        sim_time = render(False, renderer, mesh, xnew, radius, sim_time, dt)

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


run()
