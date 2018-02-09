import matplotlib.pyplot as plt
import numpy as np
import random


class Plotter:
    
    def Start(self, max_iter):
        
        self.x = np.linspace(0,1,max_iter)
        self.y = np.zeros(max_iter)
        
        plt.ion() #interactive ON

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.set_ylim(bottom=0, top=1)
        
        self.line, = ax.plot(self.x, self.y, 'ro')
    

    def Update(self, cost):
        self.y[i] = cost
        self.line.set_ydata(self.y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
plotter = Plotter()
plotter.Start(100)
for i in range(100):
    plotter.Update(random.random())
