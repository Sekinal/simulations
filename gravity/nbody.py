import taichi as ti
import math

# Initialize Taichi with CPU support
ti.init(arch=ti.cpu)

# Constants
G = 6.67430e-11  # Gravitational constant
num_bodies = 500  # Number of smaller bodies
dt = 1e6  # Time step
softening = 1e3  # Softening parameter

# Disk parameters
disk_radius = 5e10  # Radius of the disk
min_radius = 1e10   # Minimum radius from center

# Central body parameters
central_mass = 1e25  # Mass of the central body (much larger than disk particles)
total_bodies = num_bodies + 1  # Add one for the central body

# Define fields for positions, velocities, masses, and accelerations
pos = ti.Vector.field(2, dtype=ti.f32, shape=total_bodies)
vel = ti.Vector.field(2, dtype=ti.f32, shape=total_bodies)
mass = ti.field(dtype=ti.f32, shape=total_bodies)
acc = ti.Vector.field(2, dtype=ti.f32, shape=total_bodies)

@ti.kernel
def initialize():
    # Initialize central body (at index 0)
    pos[0] = ti.Vector([0.0, 0.0])  # Center position
    vel[0] = ti.Vector([0.0, 0.0])  # Stationary
    mass[0] = central_mass

    # Initialize disk particles
    for i in range(1, total_bodies):
        # Generate random angle and radius for polar coordinates
        theta = ti.random() * 2.0 * math.pi
        radius = ti.sqrt(ti.random()) * (disk_radius - min_radius) + min_radius
        
        # Convert polar to Cartesian coordinates
        pos[i] = ti.Vector([
            radius * ti.cos(theta),
            radius * ti.sin(theta)
        ])
        
        # Calculate orbital velocity for circular orbit around central mass
        orbital_speed = ti.sqrt(G * central_mass / radius)
        vel[i] = ti.Vector([
            -orbital_speed * ti.sin(theta),
            orbital_speed * ti.cos(theta)
        ])
        
        # Set masses for disk particles
        mass[i] = ti.random() * 1e20 + 1e19

@ti.kernel
def compute_accelerations():
    # Reset accelerations
    for i in range(total_bodies):
        acc[i] = ti.Vector([0.0, 0.0])

    # Compute pairwise gravitational forces
    for i in range(total_bodies):
        for j in range(total_bodies):
            if i != j:
                r = pos[j] - pos[i]
                dist = r.norm() + softening
                gravitational_force = G * mass[i] * mass[j] / (dist ** 3) * r
                acc[i] += gravitational_force / mass[i]

@ti.kernel
def update():
    for i in range(total_bodies):
        # Update velocities and positions
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt

# Set up the GUI
window_size = 800
gui = ti.GUI("2D Disk with Central Body", res=(window_size, window_size))

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
        pos_norm = (pos_np / (disk_radius * 1.5)) + 0.5

        # Clear screen
        gui.clear(0x112F41)

        # Draw disk particles (indices 1 and above)
        gui.circles(pos_norm[1:], radius=2, color=0xFFFFFF)
        
        # Draw central body (larger and different color)
        gui.circle(pos_norm[0], radius=10, color=0xFF9933)  # Orange color
        
        gui.show()

if __name__ == "__main__":
    run_simulation()