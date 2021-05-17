import numpy as np
import matplotlib.pyplot as plt


def visualize_velocity(flow , x1 , x2 , name, use_liq = False):
    if use_liq: # runlic and tf 2.2 dont fit together this fix is bad
        from licpy.lic import runlic
        from licpy.plot import grey_save
        two, size_x, size_y = flow.shape
        tex = runlic(flow[1], flow[0], 15, False);
        grey_save(name, tex);
    else:
        fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
        np.random.seed(seed=12345238)
        seed_points = np.random.rand(100,2)
        ax1.streamplot(x1, x2, flow[0], flow[1], density=1.5, cmap='autumn', start_points=seed_points)
        ax1.axis('equal')
        ax1.set(xlim=(0, 1), ylim=(0, 1))
        #fig1.colorbar(stream.lines)
        fig1.savefig(name)
        plt.close(fig1)
    return

def visualize_vorticity(vorticity , x1 , x2 , name):
    fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
    c = ax1.pcolormesh(x1, x2, vorticity, cmap='PuOr', vmin=1, vmax=-1)
    fig1.colorbar(c, ax=ax1)
    fig1.savefig(name)
    plt.close(fig1)
    return
