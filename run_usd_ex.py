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
    usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "simple_env.usd"))
    
    parent_prim = usd_stage.GetPrimAtPath('/World')
    points, faces = usd_utils.children_as_mesh(usd_stage, parent_prim)

    stage = os.path.join(os.path.dirname(__file__), "warpcrowd_output.usd")
    renderer = render_usd.UsdRenderer(stage, up_axis='y')

    wc = WarpCrowd()
    wc.demo_agents(m=20, n=20)


    # faces = usd_utils.triangulate(faces, points)

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([[-30.8,0.01582,0.0]])

    radius = wc.radius

    start = time.time()
    sim_time = 0
    for i in range(700):
        wc.compute_step()
        sim_time = render(False, renderer, wc.mesh, wc.xnew_wp, radius, sim_time, wc.dt)
    print(time.time()-start)
    renderer.save()

def render(is_live, renderer, mesh, positions, sim_margin, sim_time, dt):
    
    with wp.ScopedTimer("render", detailed=False):
        time = 0.0 if is_live else sim_time 
        
        renderer.begin_frame(time)

        # box_path = renderer.root.GetPath().AppendChild('boxes')
        # box = UsdGeom.Cube.Get(renderer.stage, box_path)
        # if not box:
        #     box = UsdGeom.Cube.Define(renderer.stage, box_path)
        #     renderer._usd_add_xform(box)

        # xform = UsdGeom.Xform(xform)
        # xform_ops = xform.GetOrderedXformOps()

        # xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
        # xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
        # xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)

        renderer.render_mesh(name="mesh", points=mesh.points.numpy(), indices=mesh.indices.numpy())
        renderer.render_points(name="points", points=positions.numpy(), radius=sim_margin)
        # render_box(self, name: str, pos: tuple, rot: tuple, extents: tuple)
        renderer.end_frame()

    sim_time += dt
    return sim_time

run_class()