import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import numpy as np
import json

class gridWorld(object):
    def __init__(self, file = "gridworlds/tiny.json"):
        self.load_from_file(file)

    def __repr__(self):
        string = ""
        for y in range(self.board_mask.shape[0]):
            for x in range(self.board_mask.shape[1]):
                if self.current_state == (y, x):
                    string += 'o\t'
                else:
                    string += "{} \t".format(self.board_mask[(y, x)])
            string += "\n"
        return string
        
    def init(self, state = None):
        if (type(state) == type(None)):
            x = np.random.choice(self.board_mask.shape[1])
            y = np.random.choice(self.board_mask.shape[0])
            self.current_state = (y, x)
        else:
            self.current_state = state
        return self.state()
        
        
    def actions(self, state = None):
        if type(state) == type(None):
            state = self.current_state
        if self.terminal[state] > 0:
            return []
        return ['U', 'D', 'L', 'R']
    
    def states(self):
        for y in range(self.board_mask.shape[0]):
            for x in range(self.board_mask.shape[1]):
                if self.board_mask[y, x] == 0:
                    yield (y, x)
    
    def reward(self, state = None):
        if type(state) == type(None):
            state = self.current_state
        return self.rewards[state]
    
    def state(self):
        return self.current_state
    
    def transition_probability(self, s_next, s, a):
        if self.terminal[s] > 0:
            return 0
        
        if(a == 'D'):
            straight = (s[0] + 1, s[1])
            right = (s[0], s[1] + 1)
            left = (s[0], s[1] - 1)
        elif(a == 'U'):
            straight = (s[0] - 1, s[1])
            right = (s[0], s[1] - 1)
            left = (s[0], s[1] + 1)
        elif(a == 'R'):
            straight = (s[0], s[1] + 1)
            right = (s[0] + 1, s[1])
            left = (s[0] - 1, s[1])
        elif(a == 'L'):
            straight = (s[0], s[1] - 1)
            right = (s[0] - 1, s[1])
            left = (s[0] + 1, s[1])
        
        if s_next == straight and s_next in self.states():
            return self.p
        
        if (s_next == right or s_next == left) and s_next in self.states():
            return (1 - self.p)/2
        
        if (s_next == s):
            p = 0
            p += self.p if straight not in self.states() else 0
            p += (1 - self.p)/2 if right not in self.states() else 0
            p += (1 - self.p)/2 if left not in self.states() else 0
            return p
        return 0
        
    
    def step(self, action):
        states = []
        probs = []
        idx = []
        i = 0
        for s_next in self.states():
            if self.transition_probability(s_next, self.current_state, action) > 0:
                idx.append(i)
                i+= 1
                states.append(s_next)
                probs.append(self.transition_probability(s_next, self.current_state, action))
                
        self.current_state = states[np.random.choice(idx, p = probs)]  

        
    def render(self, show_reward = True, show_state = True, show_terminal = True, show = True):
        # create discrete colormap
        cmap = colors.ListedColormap(['white', 'gray'])
        
        fig, ax = plt.subplots()
        ax.imshow(self.board_mask, cmap=cmap)
        
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, self.board_mask.shape[1] , 1));
        ax.set_yticks(np.arange(-.5, self.board_mask.shape[0] , 1));
        
        patches = []
        
        if show_terminal:
            for x in range(self.terminal.shape[1]):
                for y in range(self.terminal.shape[0]):
                    if self.terminal[(y, x)] > 0:
                        patches.append(Rectangle((x-0.45, y-0.45), 0.9, 0.9, edgecolor = 'k', fill = False))
        
        if show_state:
            patches.append(plt.Circle((self.current_state[1], self.current_state[0]), radius=0.2, color='b'))
        
        if show_reward:
            for x in range(self.rewards.shape[1]):
                for y in range(self.rewards.shape[0]):
                    if self.board_mask[(y, x)] == 0:
                        ax.annotate(self.rewards[(y, x)], (x-0.4,y+0.4))
        
        for patch in patches:
            ax.add_patch(patch)
            
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        if show:
            plt.show()
            
        return fig
        
    def load_from_file(self, filename):
        with open(filename) as file:
            data = json.load(file)
        self.current_state = (data['initial_state'][0], data['initial_state'][1])
        self.board_mask = np.array(data['board_mask'])
        self.terminal = np.array(data['terminal'])
        self.rewards = np.array(data['rewards'])
        self.p = data['probability']