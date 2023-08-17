import numpy as np
from pxr import UsdGeom, Gf, Usd


def children_as_mesh(stage, parent_prim):
    children = parent_prim.GetAllChildren()
    children = [child.GetPrimPath() for child in children]
    points, faces = get_mesh(stage, children)
    return points, faces

def get_mesh(usd_stage, objs):

    points, faces = [],[]

    for obj in objs:
        f_offset = len(points)
        f, p = convert_to_mesh(usd_stage.GetPrimAtPath(obj))
        if len(p) == 0: continue
        points.extend(p)
        faces.extend(f+f_offset)

    return points, faces

def convert_to_mesh(prim):
    ''' convert a prim to BVH '''

    # Get mesh name (prim name)
    m = UsdGeom.Mesh(prim)

    # Get verts and triangles
    tris = m.GetFaceVertexIndicesAttr().Get()

    if not tris:
        return [], []

    tris_cnt = m.GetFaceVertexCountsAttr().Get()

    verts = m.GetPointsAttr().Get()

    tri_list = np.array(tris)
    vert_list = np.array(verts)

    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotationMatrix()
    # rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    rotation = rotation.GetOrthonormalized()

    # New vertices
    vert_list = np.dot((vert_list * scale ), rotation) + translation

    tri_list = convert_to_triangle_mesh(tris, tris_cnt)
    # faces = parse_faces(tris, tris_cnt)
    # tri_list = convert_to_triangle_mesh(faces)
    tri_list = tri_list.flatten()
    # tri_list = triangulate(tris_cnt, tris)

    return tri_list, vert_list


def triangulate(face_counts, face_indices):
    '''
    Taken from :
    https://github.com/NVIDIA/warp/blob/main/warp/tests/test_mesh_query_point.py#L248
    '''

    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0

    for nb in face_counts:
        for i in range(nb - 2):
            tri_indices[ctr] = face_indices[wedgeIdx]
            tri_indices[ctr + 1] = face_indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb

    return tri_indices





def convert_to_triangle_mesh(FaceVertexIndices, FaceVertexCounts):
    """
    Convert a list of vertices and a list of faces into a triangle mesh.
    
    A list of triangle faces, where each face is a list of indices of the vertices that form the face.
    """
    
    # Parse the face vertex indices into individual face lists based on the face vertex counts.

    faces = []
    start = 0
    for count in FaceVertexCounts:
        end = start + count
        face = FaceVertexIndices[start:end]
        faces.append(face)
        start = end

    # Convert all faces to triangles
    triangle_faces = []
    for face in faces:
        if len(face) < 3:
            newface = []  # Invalid face
        elif len(face) == 3:
            newface = [face]  # Already a triangle
        else:
            # Fan triangulation: pick the first vertex and connect it to all other vertices
            v0 = face[0]
            newface = [[v0, face[i], face[i + 1]] for i in range(1, len(face) - 1)]

        triangle_faces.extend(newface)
    
    return np.array(triangle_faces)

