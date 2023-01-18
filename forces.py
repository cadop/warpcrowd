from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
import warp as wp

Tau = wp.constant(0.5)
A = wp.constant(2000.0)
B = wp.constant(0.08)
kn = wp.constant(1.2 * 100000)
kt = wp.constant(2.4 * 100000)

max_speed = wp.constant(100.0)

v_desired = wp.constant(2.5)


@wp.kernel
def integrate(x : wp.array(dtype=wp.vec3),
                v : wp.array(dtype=wp.vec3),
                f : wp.array(dtype=wp.vec3),
                dt: float,
                xnew: wp.array(dtype=wp.vec3),
                vnew: wp.array(dtype=wp.vec3), 
            ):
    
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]

    v1 = v0 + (f0*1.0) * dt
    x1 = x0 + v1 * dt

    xnew[tid] = x1
    vnew[tid] = v1

@wp.func
def calc_wall_force(rr_i: wp.vec3,
                    ri: float,
                    vv_i: wp.vec3,
                    mesh: wp.uint64):
    '''
    rr_i : position
    ri : radius
    vv_i : velocity
    Computes: (A * exp[(ri-diw)/B] + kn*g(ri-diw))*niw - kt * g(ri-diw)(vi * tiw)tiw
    '''

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    # Define the up direction 
    up_dir = wp.vec3(0.0, 0.0, 1.0)

    max_dist = float(ri * 100.0)

    has_point = wp.mesh_query_point(mesh, rr_i, max_dist, sign, face_index, face_u, face_v)

    if (not has_point):
        return wp.vec3(0.0, 0.0, 0.0)
        
    p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    # d_iw = distance to wall W
    d_iw = wp.length(p - rr_i)

    # vector of the wall to the agent
    nn_iw = wp.normalize(rr_i - p)
    # perpendicular vector of the agent-wall (tangent force)
    tt_iw = wp.cross(up_dir, nn_iw)
    if wp.dot(vv_i, tt_iw) < 0.0: 
        tt_iw = -1.0 * tt_iw

    # Compute force
    #  (A * exp[(ri-diw)/B] + kn*g(ri-diw))*niw - kt * g(ri-diw)(vi * tiw)tiw
    f1 = (A * wp.exp((ri-d_iw)/B) ) + kn
    force = ( f1 * G(ri,d_iw)*nn_iw) - ((kt*G(ri,d_iw)) * wp.dot(vv_i,tt_iw) * tt_iw)

    return force


@wp.func
def calc_agent_force(rr_i: wp.vec3, 
                     ri: float, 
                     vv_i: wp.vec3, 
                     pn_rr: wp.array(dtype=wp.vec3), 
                     pn_vv: wp.array(dtype=wp.vec3), 
                     pn_r: wp.array(dtype=float),
                     pn: float,
                     grid : wp.uint64,
                     ):

    #  Sum the forces of neighboring agents 
    force = wp.vec3(0.0, 0.0, 0.0)

    #  Set the total force of the other agents to zero
    ff_ij = wp.vec3(0.0, 0.0, 0.0)

    rr_j = wp.vec3(0.0, 0.0, 0.0)

    # rj = 0.0

    #  Iterate through the neighbors and sum (f_ij)
    # for j in range(5):

    # create grid query around point
    query = wp.hash_grid_query(grid, rr_i, pn)
    index = int(0)

    while(wp.hash_grid_query_next(query, index)):
        
        j = index

        neighbor = pn_rr[j]

        # compute distance to neighbor point
        dist = wp.length(rr_i-neighbor)
        if (dist <= pn):
            
            rr_j = pn_rr[j]

            #  Get position and velocity of neighbor agent
            vv_j = pn_vv[j]

            #  Get radii of neighbor agent
            rj = pn_r[j]

            #  Pass agent position to AgentForce calculation
            ff_ij = agent_force(rr_i, ri, vv_i, rr_j, rj, vv_j)

            #  Sum Forces
            force += ff_ij
    
    return force

@wp.func
def calc_goal_force(goal: wp.vec3, 
                    pos: wp.vec3, 
                    vel: wp.vec3, 
                    mass: float, 
                    v_desired: float, 
                    dt: float):
    return goal_force(goal, pos, vel, mass, v_desired, dt)

@wp.func
def agent_force(rr_i: wp.vec3, 
                ri: float, 
                vv_i: wp.vec3, 
                rr_j: wp.vec3, 
                rj: float, 
                vv_j: wp.vec3):
    # (Vector3 rr_i, float ri, Vector3 vv_i, Vector3 rr_j, float rj, Vector3 vv_j)

    #  Calculate the force exerted by another agent
    #  Take in this agent (i) and a neighbors (j) position and radius

    #  Calculate rij - dij
    #  Sum of radii
    rij = ri + rj
    #  distance between center of mass
    d_ij = wp.length(rr_i - rr_j)

    #  "n_ij is the normalized vector points from pedestrian j to i"
    n_ij = wp.normalize(rr_i - rr_j) # Normalized vector pointing from j to i

    #  t_ij "Vector of tangential relative velocity pointing from i to j." 
    #  A sliding force is applied on agent i in this direction to reduce the relative velocity.
    t_ij = vv_j - vv_i
    deltaV = wp.dot(vv_j - vv_i, t_ij)

    #  Calculate f_ij
    force = ( ( proximity(rij, d_ij) + repulsion(rij, d_ij) ) * n_ij ) +  sliding(rij, d_ij, deltaV, t_ij)

    return force

@wp.func
def goal_force(goal: wp.vec3, 
                i_xyz: wp.vec3, 
                v_i: wp.vec3, 
                m_i: float, 
                v_desired: float, 
                dt: float):

    ee_i = wp.normalize(goal - i_xyz)
    force = m_i * ( ( (v_desired * ee_i) - v_i ) / (dt) ) #  alt is to replace dt with  Parameters.T 

    return force 

@wp.func
def G(r_ij: float, 
      d_ij: float
      ):
    # g(x) is a function that returns zero if pedestrians touch
    # otherwise is equal to the argument x 
    if (d_ij > r_ij): return 0.0
    return r_ij - d_ij

@wp.func
def proximity(r_ij: float, 
             d_ij: float):
    force = A * wp.exp( (r_ij - d_ij) / B)
    return force

@wp.func
def repulsion(r_ij: float, 
              d_ij: float
              ):
    force = kn * G(r_ij, d_ij)
    return force 

@wp.func
def sliding(r_ij: float, 
            d_ij: float, 
            deltaVelocity: float, 
            t_ij: wp.vec3):
    force = kt * G(r_ij, d_ij) * (deltaVelocity * t_ij)
    return force

@wp.func
def compute_force(rr_i: wp.vec3, 
                    ri: float,
                    vv_i: wp.vec3, 
                    mass:float,
                    goal:wp.vec3, 
                    pn_rr: wp.array(dtype=wp.vec3),
                    pn_vv: wp.array(dtype=wp.vec3),
                    pn_r: wp.array(dtype=float), 
                    dt: float,
                    pn: float,
                    grid : wp.uint64,
                    mesh: wp.uint64
                    ):
    ''' 
    # agent is a position
    rr_i : position
    ri : radius
    vv_i : velocity
    pn_rr : List[perceived neighbor positions]
    pn_vv : List[perceived neighbor velocities]
    pn_r : List[perceived neighbor radius]
    '''

    # Get the force for this agent to the goal
    goal = calc_goal_force(goal, rr_i, vv_i, mass, v_desired, dt)
    agent = calc_agent_force(rr_i, ri, vv_i, pn_rr, pn_vv, pn_r, pn, grid)

    wall = calc_wall_force(rr_i, ri, vv_i, mesh)

    force = goal + agent + wall

    force = wp.normalize(force) * wp.min(wp.length(force), max_speed)

    return force


