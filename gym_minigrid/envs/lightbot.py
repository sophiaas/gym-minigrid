from gym_minigrid.minigrid import *
from gym_minigrid.register import register
# import torch
import pickle

def gen_fractal(stage, primitive=None, mask=None, padding=2, portion=None):
    if primitive is None:
        primitive = np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ])
    if mask == None:
        mask = primitive
    shape = primitive.shape
    for stage in range(stage):
        
        tiled = np.tile(primitive, shape)
        mask = np.repeat(np.repeat(primitive, shape[0], axis=0), shape[1], axis=1)
        new_primitive = tiled * mask
        primitive = new_primitive
    if portion:
        xs = portion['xs']
        ys = portion['ys']
        unit = int(len(primitive)/3)
        primitive = primitive[xs[0]*unit:xs[1]*unit, ys[0]*unit:ys[1]*unit]
    size = list(np.array(primitive.shape) + padding * 2)
    x, y = np.nonzero(primitive)
    idxs = [[a+padding, b+padding] for a, b in zip(x,y)]
    return size, idxs

primitives = {
    'cross': np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ]),
    'square': np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ]),
    'triangle': np.array([
            [0,1,0],
            [0,0,0],
            [1,0,1]
        ])
}

puzzles = {}
for i in range(4):
    for p in primitives.keys():
        size, idxs = gen_fractal(i, primitives[p])
        puzzles['fractal_'+p+'_'+str(i)] = {'size': size, 'light_idxs': idxs}
        
portions = [
    {'xs': [1,2], 'ys': [0,2]},
    {'xs': [1,3], 'ys': [0,2]},
    {'xs': [1,3], 'ys': [0,3]}
]
for i in range(1, 4):
    for j, p in enumerate(portions):
        size, idxs = gen_fractal(i, primitives['cross'], portion=p)
        puzzles['fractal_cross_'+str(i-1)+'-'+str(j)] = {'size': size, 'light_idxs': idxs}   
        
puzzles['test'] = {'size': 9, 'light_idxs': [[5,5]]}

class LightbotEnv(MiniGridEnv):
    
    class Actions(IntEnum):
        toggle = 0
        jump = 1
        forward = 2
        right = 3
        left = 4

    def __init__(self, config):
        self.config = config
        self.agent_start_pos = config.agent_start_pos
        self.agent_start_dir = config.agent_start_dir
        self.episode = 0
        
        size = puzzles[config.puzzle_name]['size']
        if type(size) == list:
            width, height = size
        else:
            height, width = size, size
            
        print('size {}'.format(size))
            
        self.light_idxs = puzzles[config.puzzle_name]['light_idxs']
        self.reward_fn = [float(x) for x in config.reward_fn.split(',')]
        self.toggle_ontop = config.toggle_ontop
        self.name = 'lightbot_minigrid'
        
        super().__init__(
            width = width,
            height = height,
            max_steps=config.max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            custom_actions=True
        )
        
        self.actions = LightbotEnv.Actions
        self.action_space = spaces.Discrete(5)        
        print('action space: {}'.format(self.action_space))


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.num_lights = len(self.light_idxs)
        self.lights_on = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i in self.light_idxs:
            self.grid.set(i[0], i[1], Light())


        # Place the agent
        print('agent start pos: {}'.format(self.agent_start_pos))
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "turn on all of the lights"
    
    def make_move(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        curr_pos = self.agent_pos
        
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        curr_cell = self.grid.get(*curr_pos)   
        
        reward = self.reward_fn[-1]
        done = False
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

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if self.toggle_ontop:
                if curr_cell != None and curr_cell.type == 'light' and not curr_cell.is_on:
                    success = curr_cell.toggle()
                    self.lights_on += 1
                    reward = self.reward_fn[1]
            else:
                if fwd_cell != None and fwd_cell.type == 'light' and not fwd_cell.is_on:
                    success = fwd_cell.toggle()
                    self.lights_on += 1
                    reward = self.reward_fn[1]

        elif action == self.actions.jump:
            reward = self.reward_fn[-1]      

        else:
            assert False, "unknown action"
            
        if self.lights_on == self.num_lights:
            reward = self.config.reward_fn[0]
            done = True
        return reward, done
    
    def get_data(self):
        data = {
            'coords': self.agent_pos,
            'direction': self.agent_dir,
            'light': 1 if self.grid.get(*self.agent_pos).type == 'light' else 0,
            'light_on': 1 if self.grid.get(*self.agent_pos).is_on else 0
        }
        return data

    def step(self, action):
        reward = self.reward_fn[-1]
        done = False     

        reward, done = self.make_move(action)
        self.step_count += 1
        frame_update = 1

        if self.step_count >= self.max_steps:
            done = True
        if done:
            self.episode += 1
        obs = self.gen_obs()
        print('agent_pos: {}'.format(self.agent_pos))
        data = self.get_data()
        return obs, reward, done, data
    
    def get_num_actions(self):
        return self.action_space.n
    
    def get_obs_dim(self):
        return self.observation_space.shape
    
    def get_frame_count(self):
        return self.step_count
    
    def _seed(self, seed):
        np.random.seed(seed)
    
    def get_curr_pos(self):
        return self.agent_pos


register(
    id='MiniGrid-Lightbot-v0',
    entry_point='gym_minigrid.envs:Lightbot'
)
