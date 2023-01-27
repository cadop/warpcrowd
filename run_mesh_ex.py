import time
import warp as wp
import numpy as np

import mesh_utils
from warpforce import WarpCrowd

# wp.config.mode = "debug"
# wp.config.print_launches = True
# wp.config.verify_cuda = True

#### Initialize WARP #####
wp.init()

def run_class():

    points, faces = mesh_utils.load_obj('test.obj')

    wc = WarpCrowd()
    # Setup Demo Agents using:
    # wc.demo_agents()

    # Alternatively, specifiy agents directly and call configure_params
    m,n,s = 30,30,1.1
    wc.agents_pos = np.asarray([
                            np.array([(s/2) + (x * s), (s/2) + (y * s), 0], dtype=np.double) 
                            for x in range(m) 
                            for y in range(n)
                           ])
    wc.nagents = len(wc.agents_pos)
    wc.configure_params() # Call to setup params for warp, use after defining agents

    wc.config_hashgrid()
    wc.config_mesh(np.asarray(points), np.asarray(faces))
    wc.update_goals([[-30.8,0.01582,0.0]])

    start = time.time()
    for i in range(750):
        wc.compute_step()
        p = wc.xnew_wp.numpy()
    print(time.time()-start)

run_class()
