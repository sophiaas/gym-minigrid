from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class LightbotEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        light_idxs=None,
        agent_start_pos=None,
        agent_start_dir=None,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.light_idxs = light_idxs

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        if self.light_idxs is None:
            self.light_idxs = [(int(width/2), int(height/2))]
    
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        for i in self.light_idxs:
            self.grid.set(i[0], i[1], Light())


        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "turn on all of the lights"

register(
    id='MiniGrid-Lightbot-v0',
    entry_point='gym_minigrid.envs:LightbotEnv'
)