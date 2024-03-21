import numpy as np

def lj_force(r, epsilon, sigma):
    """
    r: distance between two particles
    epsilon: depth of the potential well
    sigma: distance at which the inter-particle potential is zero
    
    
    returns: magnitude of the force
    -> force
    
    
    Explanation:
    Lennard-Jones force function.
    """
    
    r = np.asarray(r)
    epsilon = np.asarray(epsilon)
    sigma = np.asarray(sigma)
    
    
    return 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r



def apply_periodic_boundary_conditions(position, box_size):
    """
    position: position of a particle
    box_size: size of the simulation box
    
    
    returns: new position inside the box
    -> position
    
    
    Explanation:
    Apply periodic boundary conditions to keep the particle in the simulation box.
    """
    return position - np.rint(position / box_size) * box_size





def velocity_verlet_full_step(positions, velocities, masses, epsilon, sigma, dt, box_size):
    """
    positions: positions of all particles
    velocities: velocities of all particles
    masses: masses of all particles
    epsilon: depth of the potential well
    sigma: distance at which the inter-particle potential is zero
    dt: time step
    box_size: size of the simulation box
    
    
    returns: new positions and velocities after one time step
    -> positions, velocities
    
    
    Explanation:
    Perform one time step of the Velocity Verlet algorithm with periodic boundary conditions.
    """
    num_particles = len(positions)
    
    # Calculate forces
    forces = np.zeros_like(positions, dtype=np.float64) # force = [x, y]
    for i in range(num_particles): #for each particle
        for j in range(i + 1, num_particles): #for each particle pair (not including previous pairs)
            r_ij = positions[j] - positions[i] #vector from particle i to particle j
            r_ij -= np.rint(r_ij / box_size) * box_size  # Apply periodic boundary conditions
            
            dist = np.linalg.norm(r_ij) #eucledian distance between particle i and j
            
            # Lennard-Jones force
            force_ij = lj_force(dist, epsilon, sigma) * r_ij / dist #x and y components of force between particle i and j
            
            x_force_ij = force_ij[0] #x component of force
            y_force_ij = force_ij[1] #y component of force
            
            
            forces[i][0], forces[i][1] = forces[i][0] - x_force_ij, forces[i][1] - y_force_ij #change signs of force components
            forces[j][0], forces[j][1] = forces[j][0] + x_force_ij, forces[j][1] + y_force_ij #change signs of force components
    
    
    # Update positions
    positions += velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2 #update positions using velocity verlet algorithm (x(t+dt) = x(t) + v(t)dt + 1/2a(t)dt^2)
    
    
    # Apply periodic boundary conditions to new positions
    positions = apply_periodic_boundary_conditions(positions, box_size)
    
    
    # Calculate new forces
    new_forces = np.zeros_like(positions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_ij = positions[j] - positions[i]
            r_ij -= np.rint(r_ij / box_size) * box_size  # Apply periodic boundary conditions
            dist = np.linalg.norm(r_ij)
            
            # LJ force
            force_ij = lj_force(dist, epsilon, sigma) * r_ij / dist #x and y components of force between particle i and j
            
            x_force_ij = force_ij[0]
            y_force_ij = force_ij[1]
            
            new_forces[i][0], new_forces[i][1] = new_forces[i][0] - x_force_ij, new_forces[i][1] - y_force_ij #change signs of force components
            new_forces[j][0], new_forces[j][1] = new_forces[j][0] + x_force_ij, new_forces[j][1] + y_force_ij #change signs of force components
    
    
    # Update velocities
    velocities += 0.5 * (forces + new_forces) / masses[:, np.newaxis] * dt
    
    
    return positions, velocities


def velocity_verlet_half_step(positions, velocities, masses, epsilon, sigma, dt, box_size):
    """
    positions: positions of all particles
    velocities: velocities of all particles
    masses: masses of all particles
    epsilon: depth of the potential well
    sigma: distance at which the inter-particle potential is zero
    dt: time step
    box_size: size of the simulation box
    
    
    returns: new positions and velocities after one time step
    -> positions, velocities
    
    
    Explanation:
    Perform one time step of the Velocity Verlet algorithm with periodic boundary conditions.
    """
    
    num_particles = len(positions)
    
    # Calculate forces
    forces = np.zeros_like(positions) # force = [x, y]
    for i in range(num_particles): #for each particle
        for j in range(i + 1, num_particles): #for each particle pair (not including previous pairs)
            r_ij = positions[j] - positions[i] #vector from particle i to particle j
            r_ij -= np.rint(r_ij / box_size) * box_size  # Apply periodic boundary conditions 
            dist = np.linalg.norm(r_ij) #eucledian distance between particle i and j
            
            # Lennard-Jones force
            force_ij = lj_force(dist, epsilon, sigma) * r_ij / dist #x and y components of force between particle i and j
            
            x_force_ij = force_ij[0] #x component of force
            y_force_ij = force_ij[1] #y component of force
            
            
            forces[i][0], forces[i][1] = forces[i][0] - x_force_ij, forces[i][1] - y_force_ij #change signs of force components
            forces[j][0], forces[j][1] = forces[j][0] + x_force_ij, forces[j][1] + y_force_ij #change signs of force components
    
    
    
    velocities_half = velocities + 0.5 * forces / masses[:, np.newaxis] * dt
    
    positions += velocities_half * dt
    
    new_forces = np.zeros_like(positions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_ij = positions[j] - positions[i]
            r_ij -= np.rint(r_ij / box_size) * box_size  # Apply periodic boundary conditions
            dist = np.linalg.norm(r_ij)
            
            # LJ force
            force_ij = lj_force(dist, epsilon, sigma) * r_ij / dist #x and y components of force between particle i and j
            
            x_force_ij = force_ij[0]
            y_force_ij = force_ij[1]
            
            new_forces[i][0], new_forces[i][1] = new_forces[i][0] - x_force_ij, new_forces[i][1] - y_force_ij #change signs of force components
            new_forces[j][0], new_forces[j][1] = new_forces[j][0] + x_force_ij, new_forces[j][1] + y_force_ij #change signs of force components
            
            
    velocities = velocities_half + 0.5 * new_forces / masses[:, np.newaxis] * dt
    
    return positions, velocities



def Step_until_end(initial_pos, initial_vel, masses, epsilon, sigma, dt, stop_time, box_size, half_step = False):
    
    func = velocity_verlet_half_step if half_step else velocity_verlet_full_step
    for step in range(int(stop_time / dt)):
        next_pos, next_vel = func(initial_pos, initial_vel, masses, epsilon, sigma, dt, box_size)
        initial_pos = next_pos
        initial_vel = next_vel
    
    return initial_pos, initial_vel