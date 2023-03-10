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
    wc.demo_agents(m=2, n=2)

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([-30.8,0.01582,0.0])

    radius = wc.radius

    start = time.time()
    sim_time = 0
    for i in range(700):
        wc.compute_step()
        sim_time = render(False, renderer, wc.mesh, wc.xnew_wp, radius, sim_time, wc.dt)
    print(time.time()-start)
    renderer.save()
from pxr import UsdGeom
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

# import numpy as np

# def velocity_to_quaternion(up, forward, velocity):
#     # Construct a quaternion that rotates the agent's forward direction to align with the velocity vector
#     # Assumes up, forward, and velocity are 3-element numpy arrays

#     forward_norm = np.linalg.norm(forward)
#     if forward_norm > 0:
#         forward = forward / forward_norm  # Normalize the forward vector
#     velocity_norm = np.linalg.norm(velocity)
#     if velocity_norm > 0:
#         velocity = velocity / velocity_norm  # Normalize the velocity vector
#     else: 
#         velocity = forward

#     dot = np.dot(forward, velocity) # Clip the dot product to avoid numerical instability
#     if dot == 1.0:
#         return np.array([1, 0, 0, 0])

#     # if np.isclose(dot, 1.0):
#         # If the forward and velocity vectors are already aligned, return the identity quaternion
#         # return np.array([1, 0, 0, 0])
#     else:
#         axis = np.cross(forward, velocity)
#         axis = up * np.sign(np.dot(axis, up))  # Project the axis onto the up plane
#         axis_norm = np.linalg.norm(axis)
#         if axis_norm > 0:
#             axis = axis / axis_norm  # Normalize the axis of rotation
#         else:
#             axis = up  # Use a default axis of rotation if the input is a zero vector

#         angle = np.arccos(dot)  # Calculate the angle of rotation with clipping
#         qw = np.cos(angle/2)  # Calculate the scalar component of the quaternion
#         qx, qy, qz = np.sin(angle/2) * axis  # Calculate the vector component of the quaternion
#         return np.array([qw, qx, qy, qz])

# def quaternion_to_euler(qw, qx, qy, qz):
#     # Normalize the quaternion
#     mag = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
#     qw /= mag
#     qx /= mag
#     qy /= mag
#     qz /= mag
    
#     # Calculate the rotation matrix from the quaternion
#     R = np.array([
#         [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
#         [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
#         [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
#     ])

#     # Extract the Euler angles from the rotation matrix
#     sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
#     roll = np.arctan2(R[2, 1], R[2, 2])
#     pitch = np.arctan2(-R[2, 0], sy)
#     yaw = np.arctan2(R[1, 0], R[0, 0])

#     return roll, pitch, yaw

# # Example usage:
# up = np.array([0, 0, 1])
# forward = np.array([1, 0, 0])
# velocity = np.array([1, 0, 0])

# q = velocity_to_quaternion(up,forward, velocity)
# print(q)
# roll, pitch, yaw = quaternion_to_euler(*q)
# print(roll, pitch, yaw)

# velocity = np.array([1, 1, 0])
# q = velocity_to_quaternion(up,forward, velocity)
# print(q)
# roll, pitch, yaw = quaternion_to_euler(*q)
# print(roll, pitch, yaw)

# velocity = np.array([1, 0, 0])
# q = velocity_to_quaternion(up,forward, velocity)
# print(q)
# roll, pitch, yaw = quaternion_to_euler(*q)
# print(roll, pitch, yaw)

# velocity = np.array([0, 0, 1])
# q = velocity_to_quaternion(up,forward, velocity)
# print(q)
# roll, pitch, yaw = quaternion_to_euler(*q)
# print(roll, pitch, yaw)

# velocity = np.array([0, 0, 0])
# q = velocity_to_quaternion(up,forward, velocity)
# print(q)
# roll, pitch, yaw = quaternion_to_euler(*q)
# print(roll, pitch, yaw)
