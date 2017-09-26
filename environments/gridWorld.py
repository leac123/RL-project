import numpy as np

class gridWorld(object):
    def __init__(self):
        self.rewards = np.array([[-0.04, -0.04, -0.04,     1],
                                 [-0.04, -0.04, -0.04,    -1], 
                                 [-0.04, -0.04, -0.04, -0.04]])
        
        self.board_mask = np.array([[0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0]])
        
        self.terminal = np.array([[0, 0, 0, 1],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 0]])
        
        self.state = (2, 0)
        self.p = 0.8
    def __repr__(self):
        string = ""
        for y in range(self.board_mask.shape[0]):
            for x in range(self.board_mask.shape[1]):
                if self.state == (y, x):
                    string += 'o\t'
                else:
                    string += "{} \t".format(self.board_mask[(y, x)])
            string += "\n"
        return string
        
        
    def actions(self, state = None):
        if type(state) == type(None):
            state = self.state
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
            state = self.state
        return self.rewards[state]
    
    def transition_probability(self, s_next, s, a):
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
            if self.transition_probability(s_next, self.state, action) > 0:
                idx.append(i)
                i+= 1
                states.append(s_next)
                probs.append(self.transition_probability(s_next, self.state, action))
                
        self.state = states[np.random.choice(idx, p = probs)]  

        
    def render(self):
        print(self)