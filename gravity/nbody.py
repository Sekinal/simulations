import taichi as ti
import math

# Initialize Taichi with CPU support
ti.init(arch=ti.cpu)

# Constants
G = 6.67430e-11  # Gravitational constant
num_bodies = 1000  # Number of bodies
dt = 1e4  # Time step
softening = 1e3  # Softening parameter

# Disk parameters
disk_radius = 5e10  # Radius of the disk
min_radius = 1e10   # Minimum radius from center

# Define fields for positions, velocities, masses, and accelerations
pos = ti.Vector.field(2, dtype=ti.f32, shape=num_bodies)
vel = ti.Vector.field(2, dtype=ti.f32, shape=num_bodies)
mass = ti.field(dtype=ti.f32, shape=num_bodies)
acc = ti.Vector.field(2, dtype=ti.f32, shape=num_bodies)

@ti.kernel
def initialize():
    for i in range(num_bodies):
        # Generate random angle and radius for polar coordinates
        theta = ti.random() * 2.0 * math.pi  # Random angle
        # Random radius with square root distribution for uniform density
        radius = ti.sqrt(ti.random()) * (disk_radius - min_radius) + min_radius
        
        # Convert polar to Cartesian coordinates
        pos[i] = ti.Vector([
            radius * ti.cos(theta),  # x position
            radius * ti.sin(theta)   # y position
        ])
        
        # Initialize orbital velocities
        # Calculating orbital velocity for approximate circular motion
        orbital_speed = ti.sqrt(G * 1e24 / radius)  # Simple circular orbit speed
        vel[i] = ti.Vector([
            -orbital_speed * ti.sin(theta),  # Perpendicular to radius
            orbital_speed * ti.cos(theta)
        ])
        
        # Randomize masses (smaller range for more stability)
        mass[i] = ti.random() * 1e23 + 1e22

@ti.kernel
def compute_accelerations():
    # Reset accelerations
    for i in range(num_bodies):
        acc[i] = ti.Vector([0.0, 0.0])

    # Compute pairwise gravitational forces
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                r = pos[j] - pos[i]
                dist = r.norm() + softening
                gravitational_force = G * mass[i] * mass[j] / (dist ** 3) * r
                acc[i] += gravitational_force / mass[i]

@ti.kernel
def update():
    for i in range(num_bodies):
        # Update velocities and positions
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt

# Set up the GUI
window_size = 800
gui = ti.GUI("2D Disk N-body Simulation", res=(window_size, window_size))

def run_simulation():
    initialize()

    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            break

        # Update simulation
        for _ in range(5):
            compute_accelerations()
            update()

        # Prepare positions for visualization
        pos_np = pos.to_numpy()
        # Normalize positions to [0, 1] for visualization
        pos_norm = (pos_np / (disk_radius * 1.5)) + 0.5  # Adjusted scale factor
        
        # Clear screen and draw particles
        gui.clear(0x112F41)  # Dark blue background
        gui.circles(pos_norm, radius=2, color=0xFFFFFF)
        gui.show()

if __name__ == "__main__":
    run_simulation()