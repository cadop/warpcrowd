''' Python implementation of the Predictive Avoidance Model (PAM)
    from 
    A Predictive Collision Avoidance Model for Pedestrian Simulation,
    I. Karamouzas, P. Heil, P. van Beek, M. H. Overmars
    Motion in Games (MIG 2009), Lecture Notes in Computer Science (LNCS), Vol. 5884, 2009
'''

from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
import warp as wp

# @TODO move to wp.struct

# The agents field of view
field_of_view = wp.constant( 200.0)
# The agents radius ? Used here in this implementation or in sim?
agent_radius = wp.constant( 0.5)
# Minimum agent distance
min_agent_dist = wp.constant( 0.1)
# the mid distance parameters in peicewise personal space function predictive force dist
dmid = wp.constant( 4.0)
# KSI  
ksi = wp.constant( 0.5)
# Nearest Neighbour distance ? Used here in this implementation or in sim?
neighbor_dist = wp.constant( 10.0)
# Maximum neighbours to consider ? Used here in this implementation or in sim?
max_neighbors = wp.constant( 3)
# Maximum acceleration ? Used here in this implementation or in sim/physics?
max_accel = wp.constant( 20.0)
# Maximum speed
max_speed = wp.constant( 7)
# Preferred Speed
preferred_speed = wp.constant( 2.5)
# Goal acquired radius 
goal_radius = wp.constant( 1.0)
# Time Horizon
time_horizon = wp.constant( 4.0)
# Agent Distance
agent_dist = wp.constant( 0.1)
# Wall Distance
wall_dist = wp.constant( 0.1)
# Wall Steepnes
wall_steepness = wp.constant( 2.0)
# Agent Strength
agent_strength = wp.constant( 1.0)
# wFactor, factor to progressively scale down forces in when in a non-collision state
w_factor = wp.constant( 0.8)
# Noise flag (should noise be added to the movement action)
noise = wp.constant( False )
force_clamp = wp.constant( 40.0)

## @TODO fix these constants

# # *private* Ideal wall distance
# _ideal_wall_dist = wp.constant( agent_radius + wall_dist)
# # *private* Squared ideal wall distance
# _SAFE = wp.constant( _ideal_wall_dist * _ideal_wall_dist)
# # *private* Agent Personal space
# _agent_personal_space = wp.constant( agent_radius + min_agent_dist)
# # *private* the min distance parameters in peicewise personal space function
# _dmin = wp.constant( agent_radius + _agent_personal_space)
# # *private* the max distance parameters in peicewise personal space function
# _dmax = wp.constant( time_horizon * max_speed)
# # *private* FOV cosine
# _cosFOV = wp.constant( np.cos((0.5 * np.pi * field_of_view) / 180.0))


# *private* Ideal wall distance
_ideal_wall_dist = wp.constant( 0.5 + 0.1)
# *private* Squared ideal wall distance
_SAFE = wp.constant( ( 0.5 + 0.1) * ( 0.5 + 0.1))
# *private* Agent Personal space
_agent_personal_space = wp.constant( 0.5 + 0.1)
# *private* the min distance parameters in peicewise personal space function
_dmin = wp.constant( 0.5 + 0.5 + 0.1)
# *private* the max distance parameters in peicewise personal space function
_dmax = wp.constant( 4.0 * 7)
# *private* FOV cosine
_cosFOV = wp.constant(float(np.cos((0.5 * np.pi * 200.0) / 180.0)))



@wp.kernel
def get_forces(positions: wp.array(dtype=wp.vec3),
                velocities: wp.array(dtype=wp.vec3),
                goals: wp.array(dtype=wp.vec3),
                radius: wp.array(dtype=float),
                mass: wp.array(dtype=float),
                dt: float,
                percept : wp.array(dtype=float),
                grid : wp.uint64,
                mesh: wp.uint64,
                inv_up: wp.vec3,
                forces: wp.array(dtype=wp.vec3),
                ):

    # thread index
    tid = wp.tid()

    cur_pos = positions[tid]
    cur_rad = radius[tid]
    cur_vel = velocities[tid]
    cur_mass = mass[tid]
    goal = goals[tid]
    pn = percept[tid]

    _force = compute_force(cur_pos,
                                cur_rad,
                                cur_vel,
                                cur_mass,
                                goal,
                                positions,
                                velocities,
                                radius,
                                dt,
                                pn,
                                grid,
                                mesh)

    # Clear any vertical forces with Element-wise mul
    _force = wp.cw_mul(_force, inv_up)

    # compute distance of each point from origin
    forces[tid] = _force

@wp.func
def ray_intersects_disc(pi: wp.vec3, 
                        pj: wp.vec3, 
                        v: wp.vec3, 
                        r: float
                        ):
    
    # calc ray disc est. time to collision
    t = 0.0
    w = pj - pi
    a = wp.dot(v, v)
    b = wp.dot(w, v)
    c = wp.dot(w, w) - (r * r)
    discr = (b * b) - (a * c)
    if discr > 0.0:
        t = (b - wp.sqrt(discr)) / a
        if t < 0.0:
            t = 999999.0
    else:
        t = 999999.0
    
    return t

def mag(v):
    # calc magnitude of vector
    v_mag = np.sqrt(v.dot(v))
    return v_mag 

def norm(v):
    # normalize a vector
    v_norm = np.array([0, 0, 0], dtype='float64')
    magnitude = mag(v)
    if magnitude > 0.0:
        v_norm = v / magnitude
    return v_norm

def get_neighbors(cur, 
                  agents, 
                  pn_r):
    dist = distance.cdist([cur], agents)
    pn = dist < pn_r
    # Index to remove is when its zero
    pn_self = dist == 0
    pn_self = np.nonzero(pn_self)
    pn[pn_self] = False

    pn = np.nonzero(pn)
    return pn


def wall_force(obstacles, 
               rr_i, 
               closest_point, 
               SAFE, 
               add_force):

    for i in range(len(obstacles)):
        # Step 1: get closest point on obstacle to agent
        # [[ Need python code for this in simulation ]]
        
        n_w = rr_i - closest_point
        d_w = mag(n_w) * mag(n_w)

        if (d_w < SAFE):
            d_w = np.sqrt(d_w)
            if (d_w > 0):
                n_w /= d_w
            if ((d_w - Parameters.agent_radius) < 0.001):
                dist_min_radius =  0.001
            else: 
                d_w - Parameters.agent_radius
            obstacle_force = (Parameters._ideal_wall_dist - d_w) / np.pow(dist_min_radius, Parameters.wall_steepness) * n_w
            add_force(obstacle_force)
    
@wp.func
def calc_goal_force(goal: wp.vec3, 
                    rr_i: wp.vec3, 
                    vv_i: wp.vec3):
    
    # Preferred velocity is preferred speed in direction of goal
    preferred_vel =  wp.mul(wp.normalize(goal - rr_i), preferred_speed )
    
    # Goal force, is always added
    goal_force = (preferred_vel - vv_i) / ksi

    return goal_force


@wp.func
def predictive_force(rr_i: wp.vec3, 
                     desired_vel: wp.vec3, 
                     desired_speed: float, 
                     pn_rr: wp.array(dtype=wp.vec3), 
                     pn_vv: wp.array(dtype=wp.vec3), 
                     pn_r: wp.array(dtype=float),
                     vv_i: wp.vec3,
                     pn: float,
                     grid : wp.uint64,
                     ):
    '''
    For now, we don't use a max_neighbor, or we don't care about real distances
    
    So solve in future by either having an array (or two) that can replicate a queue
    with pop() so that new values of ray_intersects_disc can be compared
    
    or do some pre-sorting that will allow max_neighbor to cutoff without wrong distances
    which would also mean just ending the force calculation on that iteration    
    
    '''
    # Handle predictive forces// Predictive forces

    # Keep track of if we ever enter a collision state
    agent_collision = bool(False)
    
    steering_force = wp.vec3(0.0,0.0,0.0)
    # will store a list of tuples, each tuple is (tc, agent)
    force_count = float(0.0)
    
    # create grid query around point
    query = wp.hash_grid_query(grid, rr_i, pn)
    index = int(0)

    for index in query:
        j = index
        neighbor = pn_rr[j]
        rr_j = neighbor
        
        vv_j = pn_vv[j]
        
        #  Get radii of neighbor agent
        rj = pn_r[j]
        
        combined_radius = _agent_personal_space + rj
        
        # compute distance to neighbor point
        w = rr_i-neighbor
        dist = wp.length(w)
        if (dist <= pn):
            agent_collision = True
            t = 0.0
        else:
            rel_dir = wp.normalize(w)
            _vnorm = wp.normalize(vv_i)
            _res = wp.dot(rel_dir, _vnorm)

            if _res < _cosFOV:
                continue
                
            tc = ray_intersects_disc(rr_i, rr_j, desired_vel - vv_j, combined_radius)
            if tc < time_horizon:
                if tc < t:
                    t = tc
                else:
                    continue
                            
                ## Later this can be used for max_neighbors
                # if num_neighbors < max_neighbors:
                #     t = tc
                # elif tc < t:
                #     t.pop()
                #     t.append((tc, j))

            # if tc is not within time horizon we continue
            else:
                continue 

        force_dir = rr_i + (desired_vel * t) - pn_rr[j] - (pn_vv[j] * t)
        force_dist = wp.length(force_dir)
        if force_dist > 0:
            force_dir =  force_dir / force_dist 
            
        collision_dist = wp.max(force_dist - agent_radius - pn_r[j], 0.0)
        
        #D = input to evasive force magnitude piecewise function
        D = wp.max( (desired_speed * t) + collision_dist, 0.001)
        
        force_mag = 0.0
        if D < _dmin:
            force_mag = agent_strength * _dmin / D
        elif D < dmid:
            force_mag = agent_strength
        elif D < _dmax:
            force_mag = agent_strength * (_dmax - D) / (_dmax - dmid)
        else:
            continue

        if agent_collision: 
            force_mag = wp.pow(1.0, force_count) * force_mag
        else:
            force_mag = wp.pow(w_factor, force_count) * force_mag
        
        force_count += 1.0
        steering_force = force_mag * force_dir

    return steering_force

def add_noise(steering_force):
    angle = np.random.uniform(0.0, 1.0) * 2.0 * np.pi
    dist = np.random.uniform(0.0, 1.0) * 0.001
    steering_force += dist * np.array([np.cos(angle),np.sin(angle),0], dtype='float64')

    return steering_force

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
    rr_i : position
    ri : radius
    vv_i : velocity
    pn_rr : List[perceived neighbor positions]
    pn_vv : List[perceived neighbor velocities]
    pn_r : List[perceived neighbor radius]
    '''


    # Get the goal force
    goal_force = calc_goal_force(goal, rr_i, vv_i)

    # Desired values if all was going well in an empty world
    desired_vel = vv_i + goal_force * dt
    desired_speed = wp.length(desired_vel)

    # Get obstacle (wall) forces
    obstacle_force = wp.vec3(0.0,0.0,0.0)
    #@TODO 
    # obstacle_force = wall_force()

    steering_force = wp.vec3(0.0,0.0,0.0)
    # Get predictive steering forces
    steering_force = predictive_force(rr_i, desired_vel, desired_speed, pn_rr, pn_vv, pn_r, vv_i, pn, grid)

    # # Add noise for reducing deadlocks adding naturalness
    # if noise:
    #     steering_force = add_noise(steering_force)

    # # Clamp driving force
    # if mag(steering_force) > force_clamp:
    #     steering_force = norm(steering_force) *  force_clamp
    
    return goal_force + obstacle_force + steering_force