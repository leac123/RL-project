import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

class pendulum():
    def __init__(self, mass = 1, length = 1, x0 = [0, 0], gravity = 9.81):
        # Internal state vector
        self.x = np.array(x0)
        
        # Simulation parameters
        self.step_size = 0.01 
        
        #internal parameters
        self.m = mass
        self.l = length
        self.g = gravity
        
        # Pendulum differential equations
        self.dx = lambda t, t_d, u : np.array([t_d, self.g/self.l*np.sin(t) + u/(self.l*self.l*self.m)])
    
    def init(self, x0):
        self.x = np.array(x0)
        return self.state()
    
    def transition(self, state, action):
        # Scipy ODE solver for solving ODE
        f = lambda y, t : self.dx(*y, action)
        return odeint(f, state, [0, self.step_size])[1, :]
    
    def step(self, action):
        self.x = self.transition(self.x, action)
        return self.state()
    
    def render(self, cancel = False):
        try:
            self.anim.event_source.stop()
        except:
            pass
        fig = plt.figure()
        ax = plt.axes(xlim=(-2, 2), ylim=(-1.1, 1.1), aspect='equal')
        line, = ax.plot([], [], lw=2, marker='o', markersize=6)
        animate = lambda args: line.set_data([0, np.sin(self.x[0])], [0, np.cos(self.x[0])])
        self.anim = animation.FuncAnimation(fig, animate, interval=200)
        plt.show()
        return self.anim
        
    def reward(self, state = None):
        return None
    
    def actions(self, state = None):
        return None
    
    def state(self):
        return [*self.x]
        
        