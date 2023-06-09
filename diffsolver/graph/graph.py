import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import animation

import jax.numpy as jnp

class Grapher:
    def __init__(self, rk, rec_interval=10, end=10, dt=0.01):
        self.rk = rk

        self.rec_interval = rec_interval
        self.dt = dt

        self.end = end # in seconds 
        self.rec = []
        self.t = []
        self.dx = 0.5

    def run(self):
        for i in range(int(self.end/self.dt) + 1):
            self.rk.step()
            if i % self.rec_interval == 0:
                self.rec.append(self.rk.y.copy())
                self.t.append(i * self.dt)

    def plot_1d(self, dim=None):
        if dim != None:
            recs = [r[*dim, :] for r in self.rec]
        else:
            recs = self.rec
    
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(recs[i])
            ax.set_xlabel(f"time: {self.t[i]:.1f}s", fontsize = 10)

        anim = animation.FuncAnimation(fig, animate, frames=len(self.t), repeat=True, interval=100)
        plt.show()

    def plot_2d(self, cmap="YlGn", dim=None):
        if dim != None:
            recs = [r[*dim, :, :] for r in self.rec]
        else:
            recs = self.rec

        # get the min and max
        finmin, finmax = 0, 0
        for rec in recs:
            _min = jnp.min(rec) 
            _max = jnp.max(rec)
            finmin = _min if finmin > _min else finmin
            finmax = _max if finmax < _max else finmax

        fig,[ax,cax] = plt.subplots(1,2, gridspec_kw={"width_ratios":[50,1]})
        norm = matplotlib.colors.Normalize(vmin=finmin, vmax=finmax)
        cbar = matplotlib.colorbar.ColorbarBase(cmap=cmap, norm=norm, ax=cax)

        step_size = max(recs[0].shape[1]//10, recs[0].shape[0]//10)
        step_size = step_size if step_size > 1 else 1
        yticks = jnp.arange(0, recs[0].shape[0], step_size)
        xticks = jnp.arange(0, recs[0].shape[1], step_size)

        def animate(i):
            im = ax.imshow(recs[i], cmap=cmap, norm=norm, interpolation="None")
            ax.set_xlabel(f"time: {self.t[i]:.1f}s", fontsize = 10)

            if self.dx != None:
                ax.set_yticks(yticks, labels=yticks * self.dx)
                ax.set_xticks(xticks, labels=xticks * self.dx)
        anim = animation.FuncAnimation(fig, animate, frames=len(self.t), repeat=True, interval=100)
        plt.show()

