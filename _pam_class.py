
@dataclass
class Parameters:

    # The agents field of view
    field_of_view = 200.0
    # The agents radius ? Used here in this implementation or in sim?
    agent_radius = 0.5
    # Minimum agent distance
    min_agent_dist = 0.1
    # the mid distance parameters in peicewise personal space function predictive force dist
    dmid = 4.0
    # KSI  
    ksi = 0.5
    # Nearest Neighbour distance ? Used here in this implementation or in sim?
    neighbor_dist = 10.0
    # Maximum neighbours to consider ? Used here in this implementation or in sim?
    max_neighbors = 3
    # Maximum acceleration ? Used here in this implementation or in sim/physics?
    max_accel = 20.0
    # Maximum speed
    max_speed = 7
    # Preferred Speed
    preferred_speed = 2.5
    # Goal acquired radius 
    goal_radius = 1.0
    # Time Horizon
    time_horizon = 4.0
    # Agent Distance
    agent_dist = 0.1
    # Wall Distance
    wall_dist = 0.1
    # Wall Steepnes
    wall_steepness = 2.0
    # Agent Strength
    agent_strength = 1.0
    # wFactor, factor to progressively scale down forces in when in a non-collision state
    w_factor = 0.8
    # Noise flag (should noise be added to the movement action)
    noise = False 
    force_clamp = 40.0
    
    # *private* Ideal wall distance
    _ideal_wall_dist = agent_radius + wall_dist
    # *private* Squared ideal wall distance
    _SAFE = _ideal_wall_dist * _ideal_wall_dist
    # *private* Agent Personal space
    _agent_personal_space = agent_radius + min_agent_dist
    # *private* the min distance parameters in peicewise personal space function
    _dmin = agent_radius + _agent_personal_space
    # *private* the max distance parameters in peicewise personal space function
    _dmax = time_horizon * max_speed
    # *private* FOV cosine
    _cosFOV = np.cos((0.5 * np.pi * field_of_view) / 180.0)
    