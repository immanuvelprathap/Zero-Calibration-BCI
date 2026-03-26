import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spd_manifold():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Create the "Curved" Manifold Surface (Upper half of a hyperboloid)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 1, 100)
    x = 10 * np.outer(np.cos(u), np.sinh(v))
    y = 10 * np.outer(np.sin(u), np.sinh(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cosh(v))

    # Plot the manifold surface
    ax.plot_surface(x, y, z, color='cyan', alpha=0.2, edgecolors='gray', lw=0.5)

    # 2. Define the Identity Matrix (The center/reference point)
    ax.scatter([0], [0], [10], color='red', s=100, label='Identity Matrix (I)')
    ax.text(0, 0, 11, "Reference (I)", color='red', fontsize=12, fontweight='bold')

    # 3. Plot "Raw" Subject Data (Scattered far from I)
    ax.scatter([4, 5, 3], [4, 2, 5], [14, 13, 15], color='blue', s=50, label='Raw Subject Covariances')
    
    # 4. Plot "Aligned" Data (Clustered around I)
    ax.scatter([0.5, -0.5, 0.2], [0.2, 0.5, -0.3], [10.2, 10.3, 10.1], color='green', s=50, label='Aligned Data (EA)')

    # 5. Draw Arrows showing the "Alignment" transformation
    ax.quiver(4, 4, 14, -3.2, -3.5, -3.5, color='black', arrow_length_ratio=0.1, linestyle='--')

    ax.set_title("SPD Manifold: Euclidean Alignment (EA)", fontsize=15)
    ax.set_axis_off()
    plt.legend(loc='upper right')
    
    plt.savefig('riemannian_logic.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_spd_manifold()