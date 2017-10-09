import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation

class cartPendulum():
    def __init__(self, mass_cart = 1, mass_pendulum = 1, length_pendulum = 1, x0 = [0, 0, 0, 0], gravity = 9.81):
        # Internal state vector
        self.x = np.array(x0)
        
        # Simulation parameters
        self.step_size = 0.01 
        
        #internal parameters
        self.m_c = mass_cart
        self.m_p = mass_pendulum
        self.l = length_pendulum
        self.g = gravity
        
        # Mass matrix
        self.M = lambda x, t, x_d, t_d : np.array([[1, 0, 0, 0], 
                                                   [0, 1, 0, 0], 
                                                   [0, 0, self.m_c + self.m_p, self.m_p*self.l*np.cos(t)],
                                                   [0, 0, np.cos(t), self.l]])
        # Forcing matrix
        self.F = lambda x, t, x_d, t_d, u : np.array([[x_d], 
                                                      [t_d], 
                                                      [self.m_p*self.l*t_d*t_d*np.sin(t) + u], 
                                                      [self.g*np.sin(t)]])
    
    def transition(self, state, action):
        # Scipy ODE solver for solving PDE
        f = lambda y, t : np.linalg.solve(self.M(*y), self.F(*y, action)).T[0]
        state = odeint(f, state, [0, self.step_size])[1, :]
        return state
        
    
    def step(self, action):
        # Scipy ODE solver for solving PDE
        self.x = self.transition(self.x, action)
        return self.state()
    
    def render(self, cancel = False):
        if cancel:
            try:
                self.anim.event_source.stop()
            except:
                pass
        
        fig = plt.figure()
        ax = plt.axes(xlim=(-2, 2), ylim=(-1.1, 1.1), aspect='equal')
        line, = ax.plot([], [], lw=2, marker='o', markersize=6)
        rect = Rectangle([self.x[0] - 0.2, -0.1],0.4, 0.2, fill=True, color='red', ec='black')
        ax.add_patch(rect)
        
        animate = lambda args: (rect.set_xy([self.x[0] - 0.2, -0.1]), 
                                line.set_data([self.x[0], self.x[0] + np.sin(self.x[1])],[0, np.cos(self.x[1])]))
        self.anim = animation.FuncAnimation(fig, animate, interval=40)
        plt.show()
        return self.anim
            
    def reward(self, state):
        return 1 if np.abs(state[1]) < 0.5 else 0
    
    def states(self):
        return None
    
    def actions(self):
        return None
    
    def state(self):
        return self.x
    
    def init(self, x0 = [0, 0, 0, 0]):
        self.x = np.array(x0)
        return self.state()
    
    def termianl(self):
        return None
        
        

#import sympy as sp
#m_c, m_p, l, theta, x, theta_dot, x_dot, u, g= sp.symbols('m_c, m_p, l, theta, x, theta_dot, x_dot u, g')
#
#mass_matrix = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, m_c + m_p, m_p*l*sp.cos(theta)],[0, 0, sp.cos(theta), l]])
#forcing_matrix = sp.Matrix([[x_dot], [theta_dot], [m_c*l*theta_dot*theta_dot*sp.sin(theta) + u], [g*sp.sin(theta)]])
#
#M = sp.utilities.lambdify([x, theta, x_dot, theta_dot], mass_matrix.subs([(m_c, 1), (m_p, 1), (l, 1)]))
#F = sp.utilities.lambdify([x, theta, x_dot, theta_dot, u], forcing_matrix.subs([(m_c, 1), (g, 9.81), (l, 1)]))