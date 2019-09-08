from gym_minigrid.minigrid import *
from gym_minigrid.register import register
# import torch
import pickle
import copy

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

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
#         self.episode = 0
        
        size = puzzles[config.puzzle_name]['size']
        if type(size) == list:
            width, height = size
        else:
            height, width = size, size
                        
        self.light_idxs = puzzles[config.puzzle_name]['light_idxs']
        self.reward_fn = [float(x) for x in config.reward_fn.split(',')]
        self.toggle_ontop = config.toggle_ontop
        self.name = 'lightbot_minigrid'
        
        super().__init__(
            width = width,
            height = height,
            max_steps=config.max_steps,
            agent_view_size=config.agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            custom_actions=True
        )
        
        self.actions = LightbotEnv.Actions
        self.action_space = spaces.Discrete(5)   
        curr_cell = self.grid.get(*self.agent_pos) 
        self.raw_state = {'coords': copy.deepcopy(self.agent_pos),
                          'direction': copy.deepcopy(self.agent_dir),
                          'lights_on': copy.deepcopy(self.lights_on),
                          'toggle': False}

    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.start_pos is not None
        assert self.start_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.start_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Place the agent in the starting position and direction
        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        
        # Generate raw state
        self.raw_state = {'coords': copy.deepcopy(self.agent_pos),
                          'direction': copy.deepcopy(self.agent_dir),
                          'lights_on': copy.deepcopy(self.lights_on),
                          'toggle': False}

        # Return first observation
        obs = self.gen_obs()
        return obs
    

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
        pos = self.place_agent()
        self.mission = "turn on all of the lights"
    
    def make_move(self, raw_state, action):
        toggle = False
        agent_dir = copy.deepcopy(raw_state['direction'])
        agent_pos = copy.deepcopy(raw_state['coords'])
        lights_on = copy.deepcopy(raw_state['lights_on'])
        
        # Get the position in front of the agent
        fwd_pos = DIR_TO_VEC[agent_dir] + agent_pos
        fwd_cell = copy.deepcopy(self.grid.get(*fwd_pos))
        curr_cell = copy.deepcopy(self.grid.get(*agent_pos))
                
        reward = self.reward_fn[-1]
        done = False
        if action == self.actions.left:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                agent_pos = fwd_pos

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if self.toggle_ontop:
                if curr_cell != None and curr_cell.type == 'light' and not curr_cell.is_on:
                    toggle = True
                    lights_on += 1
                    reward = self.reward_fn[1]
            else:
                if fwd_cell != None and fwd_cell.type == 'light' and not fwd_cell.is_on:
                    toggle = True
                    lights_on += 1
                    reward = self.reward_fn[1]

        elif action == self.actions.jump:
            reward = self.reward_fn[-1]      

        else:
            assert False, "unknown action"
            
        if lights_on == self.num_lights:
            reward = self.reward_fn[0]
            done = True
        raw_state = {'lights_on': lights_on, 
                     'coords': agent_pos, 
                     'direction': agent_dir, 
                     'toggle': toggle}
        return raw_state, reward, done
    
    def get_data(self):
        return self.raw_state

    def step(self, action):
        reward = self.reward_fn[-1]
        done = False     

        raw_state, reward, done = self.make_move(self.raw_state, action)
        self.agent_pos = raw_state['coords']
        self.agent_dir = raw_state['direction']
        self.lights_on = raw_state['lights_on']
        
        if raw_state['toggle']:
            curr_cell = self.grid.get(*self.agent_pos)
            if self.toggle_ontop:
                curr_cell.toggle()
            else:
                fwd_pos = self.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                fwd_cell.toggle()
                
        obs = self.gen_obs()
        self.raw_state = raw_state
        return obs, reward, done, raw_state
    
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
