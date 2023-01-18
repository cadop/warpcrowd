import warp as wp
import numpy as np
import time
import forces as sf
import os
from pxr import Usd, UsdGeom

import usd_utils

wp.config.mode = "debug"
# wp.config.print_launches = True

#### Initialize WARP #####
wp.init()
# device = "cuda"
device = "cpu"

def run():

    # generate n number of agents
    nagents = 7
    # set radius
    radius = 1
    # set mass
    mass = 1
    # set pereption radius
    perception_radius = 6

    # Set agent destination
    goal = [-30.8,0.01582,0.0]

    pos = [ [0,0,0],
            [0,1,0],
            [0,2,0],
            [1,0,0],
            [1,1,0],
            [2,0,0],
            [2,2,0],
            ]

    # randomly set position
    agents_pos = np.asarray([np.array([x,x,0]) for x in range(nagents)])
    agents_pos = np.asarray(pos)

    agents_vel = np.asarray([np.array([0,0,0]) for x in range(nagents)])
    agents_radi = np.asarray([radius for x in range(nagents)])
    agents_mass = [mass for x in range(nagents)]
    agents_percept = np.asarray([perception_radius for x in range(nagents)])
    agents_goal = np.asarray([np.array(goal, dtype=float) for x in range(nagents)])

    dt = 1.0/60.0

    @wp.kernel
    def get_forces(positions: wp.array(dtype=wp.vec3),
                    velocities: wp.array(dtype=wp.vec3),
                    goals: wp.array(dtype=wp.vec3),
                    radius: wp.array(dtype=float),
                    mass: wp.array(dtype=float),
                    dt: float,
                    pn : float,
                    grid : wp.uint64,
                    mesh: wp.uint64,
                    forces: wp.array(dtype=wp.vec3),
                    ):

        # thread index
        tid = wp.tid()

        cur_pos = positions[tid]
        cur_rad = radius[tid]
        cur_vel = velocities[tid]
        cur_mass = mass[tid]
        goal = goals[tid]

        _force = sf.compute_force(cur_pos,
                                    cur_rad, 
                                    cur_vel, 
                                    cur_mass, 
                                    goal, 
                                    positions, # TODO add back perception-specific values
                                    velocities, # TODO add back perception-specific values
                                    radius, # TODO add back perception-specific values
                                    dt,
                                    pn, 
                                    grid,
                                    mesh)

        # compute distance of each point from origin
        forces[tid] = _force

    agent_force = wp.zeros(shape=nagents,device=device, dtype=wp.vec3)

    agents_pos = wp.array(agents_pos, device=device, dtype=wp.vec3)
    agents_vel = wp.array(agents_vel, device=device, dtype=wp.vec3)
    agents_goal = wp.array(agents_goal, device=device, dtype=wp.vec3)
    agents_radi = wp.array(agents_radi, device=device, dtype=float)
    agents_mass = wp.array(agents_mass, device=device, dtype=float)

    xnew = wp.zeros_like(agents_pos)
    vnew = wp.zeros_like(agents_pos)

    grid = wp.HashGrid(dim_x=nagents, dim_y=nagents, dim_z=1, device=device)

    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "simple_env.usd"))
    UsdGeom.SetStageUpAxis(usd_stage, UsdGeom.Tokens.z)

    points = []
    faces = [] 
    
    objs = [
        '/World/Plane',
     '/World/Plane_01',
      '/World/Plane_02','/World/Plane_03','/World/Plane_04',
    '/World/Plane_05','/World/Plane_06', '/World/Plane_07','/World/Plane_08','/World/Plane_09',
    '/World/Cylinder'
    ]

    points,faces = [],[]
    for obj in objs:
        f_offset = len(points)
        p, f = append_mesh(obj, usd_stage)
        points.extend(p)
        faces.extend(f+f_offset)

    points = np.asarray(points)
    faces = np.asarray(faces)

    # create collision mesh
    mesh = wp.Mesh( points=wp.array(points, dtype=wp.vec3, device=device),
                    indices=wp.array(faces, dtype=int ,device=device)
                  )

    start = time.time()
    stage = os.path.join(os.path.dirname(__file__), "warp_sf_output.usd")

    renderer = wp.render.UsdRenderer(stage, upaxis='z')
    sim_time = 0
    for i in range(60*20):
            
        grid.build(points=agents_pos, radius=radius)

        # launch kernel
        wp.launch(kernel=get_forces,
                dim=nagents,
                inputs=[agents_pos, agents_vel, agents_goal, agents_radi, 
                        agents_mass, dt, perception_radius, grid.id, mesh.id],
                outputs=[agent_force],
                device=device
                )

        wp.launch(kernel=sf.integrate,
                dim=nagents,
                inputs=[agents_pos, agents_vel, agent_force, dt],
                outputs=[xnew, vnew],
                device=device
                )
    
        agents_pos = xnew
        agents_vel = vnew

        sim_time = render(False, renderer, mesh, xnew, 1, sim_time, dt)
        # print(agents_pos)

    print(time.time()-start)

    renderer.save()

    # af = wp.array.numpy(agent_force)

    # for idx in range(af):
    #     force = af[idx]

    # Apply forces to simulation

    # Update positions and velocities
    # for i in range(nagents):
    #     agents_pos[i] = agent_bodies[i].position
    #     agents_vel[i] = agent_bodies[i].velocity

    return 

def render(is_live, renderer, mesh, positions, sim_margin, sim_time, dt):

    with wp.ScopedTimer("render", detailed=False):
        time = 0.0 if is_live else sim_time 
        
        renderer.begin_frame(time)
        renderer.render_mesh(name="mesh", points=mesh.points.numpy(), indices=mesh.indices.numpy())
        renderer.render_points(name="points", points=positions.numpy(), radius=sim_margin)
        renderer.end_frame()

    sim_time += dt
    return sim_time

def append_mesh(path, usd_stage):
    # geo = UsdGeom.Mesh(usd_stage.GetPrimAtPath(path)) 
    # points.extend(geo.GetPointsAttr().Get())
    # faces.extend(geo.GetFaceVertexIndicesAttr().Get())

    faces, points = usd_utils.convert_to_mesh(usd_stage.GetPrimAtPath(path))

    return points, faces

run()
