import numpy as np

def load_obj(filename):
    '''Load an obj file (with full path).

    Imports pywavefront within the function. 

    Parameters
    ----------
    filename : str
        full path to obj file, including .obj

    Returns
    -------
    (vertices, faces)
        two lists of vertices and corresponding faces
    '''
    import pywavefront
    scene = pywavefront.Wavefront(filename, create_materials=True, collect_faces=True)
    verts = scene.vertices
    faces = np.array(scene.mesh_list[0].faces).flatten()
    return verts, faces 


