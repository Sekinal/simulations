import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# Simulation parameters
n = 512  # grid resolution
dt = 0.003
particles_n = 1000

# Fields
field = ti.Vector.field(2, dtype=float, shape=(n, n))  # magnetic field
particles = ti.Vector.field(2, dtype=float, shape=particles_n)  # particles for visualization

@ti.kernel
def initialize():
    # Initialize particles in a circular pattern
    for i in particles:
        angle = ti.random() * 2 * np.pi
        r = ti.sqrt(ti.random()) * 0.4 * n
        particles[i] = ti.Vector([n/2 + r * ti.cos(angle),
                                n/2 + r * ti.sin(angle)])

@ti.kernel
def compute_field():
    # Compute magnetic field from current-carrying wire
    wire_pos = ti.Vector([n/2, n/2])
    current = 1000.0  # current strength
    
    for i, j in field:
        pos = ti.Vector([float(i), float(j)])
        r = pos - wire_pos
        r_mag = r.norm()
        if r_mag > 1:  # avoid division by zero
            # B field follows right-hand rule
            field[i, j] = ti.Vector([-r.y, r.x]) * current / (r_mag * r_mag)

@ti.kernel
def move_particles():
    # Move particles along field lines
    for i in particles:
        pos = particles[i]
        if pos.x < n and pos.y < n and pos.x >= 0 and pos.y >= 0:
            # Interpolate field at particle position
            index = ti.Vector([int(pos.x), int(pos.y)])
            field_at_pos = field[index]
            
            # Update position
            if field_at_pos.norm() > 0:
                field_normalized = field_at_pos.normalized()
                particles[i] += field_normalized * dt * 100

            # Reset particles that go out of bounds
            if pos.x >= n or pos.y >= n or pos.x < 0 or pos.y < 0:
                angle = ti.random() * 2 * np.pi
                r = ti.sqrt(ti.random()) * 0.4 * n
                particles[i] = ti.Vector([n/2 + r * ti.cos(angle),
                                        n/2 + r * ti.sin(angle)])

# GUI
gui = ti.GUI("Electromagnetic Field", (n, n))

initialize()
compute_field()

while gui.running:
    move_particles()
    
    # Visualize particles
    positions = particles.to_numpy()
    gui.circles(positions/n, radius=1, color=0xffffff)
    
    # Draw wire
    gui.circle(pos=(0.5, 0.5), color=0xff0000, radius=5)
    
    gui.show()