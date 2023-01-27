import numpy as np
import mesh_utils

def get_mesh(usd_stage, objs):
    
    points, faces = [],[]

    for obj in objs:
        f_offset = len(points)
        f, p = convert_to_mesh(usd_stage.GetPrimAtPath(obj))
        points.extend(p)
        faces.extend(f+f_offset)

    return points, faces

def convert_to_mesh(prim):
    '''Coverts a usd prim to a mesh
    Imports pxr UsdGeom, Gf, Usd which should only need to exist if user uses
    usd_utils

    Parameters
    ----------
    prim : UsdPrim
        _description_

    Returns
    -------
    (faces, vertices)
        list of faces and vertices for that mesh in world space
    '''

    from pxr import UsdGeom, Gf, Usd

    # Get mesh name (prim name)
    m = UsdGeom.Mesh(prim)

    # Get verts and triangles
    tris = m.GetFaceVertexIndicesAttr().Get()

    tris_cnt = m.GetFaceVertexCountsAttr().Get()

    verts = m.GetPointsAttr().Get()

    tri_list = np.array(tris)
    vert_list = np.array(verts)

    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotationMatrix()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))

    vert_rotated = np.dot(vert_list, rotation) # Rotate points

    vert_translated = vert_rotated + translation

    vert_scaled = vert_translated
    vert_scaled[:,0] *= scale[0]
    vert_scaled[:,1] *= scale[1]
    vert_scaled[:,2] *= scale[2]

    vert_list = vert_scaled

    # Check if the face counts are 4, if so, reshape and turn to triangles
    if tris_cnt[0] == 4:
        quad_list = tri_list.reshape(-1,4)
        tri_list = mesh_utils.quad_to_tri(quad_list)
        tri_list = tri_list.flatten()

    return tri_list, vert_list

