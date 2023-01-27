'''A class to manage the warp-based version of crowd simulation
'''
import numpy as np
import warp as wp
import forces as crowd_force

class WarpCrowd():
    
    def __init__(self):
        self.device = 'cuda'

        # generate n number of agents
        self.nagents = 9
        # set radius
        self.radius = 0.7
        self.radius_min = 0.5
        self.radius_max = 1.0
        self.hash_radius = 0.7 # Radius to use for hashgrid
        # set mass
        self.mass = 80
        # set pereption radius
        self.perception_radius = 6
        self.dt = 1.0/30.0

        self.goal = [0.0,0.0,0.0]

        self.inv_up_vector = wp.vec3(1.0,1.0,1.0) 
        self.inv_up_vector[2] = 0.0 # z-up

        self.demo_agents()
        self.configure_params()
        self.params_to_warp()

    def demo_agents(self, s=1.1, m=50, n=50):
        # Initialize agents in a grid for testing
        self.agents_pos = np.asarray([
                                      np.array([(s/2) + (x * s), (s/2) + (y * s), 0], dtype=np.double) 
                                      for x in range(m) 
                                      for y in range(n)
                                    ])
        self.nagents = len(self.agents_pos)

    def configure_params(self):
        self.agents_pos = np.asarray([np.array([0,0,0]) for x in range(self.nagents)])
        self.agents_vel = np.asarray([np.array([0,0,0]) for x in range(self.nagents)])
        self.agents_radi = np.random.uniform(self.radius_min, self.radius_max, self.nagents)
        self.agents_mass = [self.mass for x in range(self.nagents)]
        self.agents_percept = np.asarray([self.perception_radius for x in range(self.nagents)])
        self.agents_goal = np.asarray([np.array(self.goal, dtype=float) for x in range(self.nagents)])

        self.xnew = np.zeros_like(self.agents_pos)
        self.vnew = np.zeros_like(self.agents_vel) 

    def params_to_warp(self):
        '''Convert all parameters to warp
        '''
        self.agent_force_wp = wp.zeros(shape=self.nagents,device=self.device, dtype=wp.vec3)
        self.agents_pos_wp = wp.array(self.agents_pos, device=self.device, dtype=wp.vec3)
        self.agents_vel_wp = wp.array(self.agents_vel, device=self.device, dtype=wp.vec3)
        self.agents_goal_wp = wp.array(self.agents_goal, device=self.device, dtype=wp.vec3)
        self.agents_radi_wp = wp.array(self.agents_radi, device=self.device, dtype=float)
        self.agents_mass_wp = wp.array(self.agents_mass, device=self.device, dtype=float)
        self.agents_percept_wp = wp.array(self.agents_percept, device=self.device, dtype=float)

        self.xnew_wp = wp.zeros_like(wp.array(self.xnew, device=self.device, dtype=wp.vec3))
        self.vnew_wp = wp.zeros_like(wp.array(self.vnew, device=self.device, dtype=wp.vec3))

    def config_hasgrid(self, nagents=None):
        '''Create a hash grid based on the number of agents
            Currently assumes z up

        Parameters
        ----------
        nagents : int, optional
            _description_, by default None
        '''

        if nagents is None: nagents = self.nagents
        self.grid = wp.HashGrid(dim_x=self.nagents, dim_y=self.nagents, dim_z=1, device=self.device)

    def config_mesh(self, points, faces):
        '''Create a warp mesh object from points and faces

        Parameters
        ----------
        points : List[[x,y,z]]
            A list of floating point xyz vertices of a mesh
        faces : List[int]
            A list of integers corresponding to vertices. Must be triangle-based
        '''
        # Init mesh for environment collision
        self.mesh = wp.Mesh( points=wp.array(points, dtype=wp.vec3, device=self.device),
                            indices=wp.array(faces, dtype=int ,device=self.device)
                            )

    def update_goals(self, goal):

        self.agents_goal = np.asarray([np.array(goal, dtype=float) for x in range(self.nagents)])
        self.agents_goal_wp = wp.array(self.agents_goal, device=self.device, dtype=wp.vec3)

    def compute_step(self):
        # Rebuild hashgrid given new positions
        self.grid.build(points=self.agents_pos_wp, radius=self.hash_radius)

        # launch kernel
        wp.launch(kernel=crowd_force.get_forces,
                dim=self.nagents,
                inputs=[self.agents_pos_wp, self.agents_vel_wp, self.agents_goal_wp, self.agents_radi_wp, 
                        self.agents_mass_wp, self.dt, self.agents_percept_wp, self.grid.id, self.mesh.id,
                        self.inv_up_vector],
                outputs=[self.agent_force_wp],
                device=self.device
                )

        # Given the forces, integrate for pos and vel
        wp.launch(kernel=crowd_force.integrate,
                dim=self.nagents,
                inputs=[self.agents_pos_wp, self.agents_vel_wp, self.agent_force_wp, self.dt],
                outputs=[self.xnew_wp, self.vnew_wp],
                device=self.device
                )
    
        self.agents_pos_wp = self.xnew_wp
        self.agents_vel_wp = self.vnew_wp