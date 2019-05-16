from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class LightbotEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    class Actions(IntEnum):
        toggle = 0
        forward = 1
        left = 2
        right = 3

    def __init__(
        self,
        size=11,
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
            see_through_walls=True,
            custom_actions=True
        )
        # Action enumeration for this environment
        self.actions = LightbotEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        if self.light_idxs is None:
            self.light_idxs = [[4,5],[5,5],[6,5],[5,4],[5,6]]
            # self.light_idxs = [[int(width/2), int(height/2)]]
        self.num_lights = len(self.light_idxs)
        self.lights_on = 0

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

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        curr_pos = self.agent_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        curr_cell = self.grid.get(*curr_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Toggle/activate an object
        elif action == self.actions.toggle:
            fwd_pos = curr_cell.toggle_in_place(self, curr_pos)

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        if action == self.actions.toggle and list(self.agent_pos) in self.light_idxs:
            reward = self._reward()
            self.lights_on += 1

        if self.lights_on == self.num_lights:
            done = True

        return obs, reward, done, {}

register(
    id='MiniGrid-Lightbot-v0',
    entry_point='gym_minigrid.envs:LightbotEnv'
)
