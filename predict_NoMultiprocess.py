import torch
import numpy as np

from Functions import Step_until_end





###############################################################################################################
## Load the tensor from the .pt file                                                                         ##
###############################################################################################################

dict = torch.load(r'Array\n4rho0.2T0.65t3.pt')
# keys: ['qpl_trajectory', 'tau_short', 'tau_long']
tensor = dict['qpl_trajectory']

# tensor has the following shape and structure:
# shape: [100,3,4,4,2]

# molecule_positions = tensor[system_index][0][time][molecule_index] is a 2D array with the x and y coordinates of the molecule
# molecule_velocities = tensor[system_index][1][time][molecule_index] is a 2D array with the x and y velocities of the molecule


masses = (1,1,1,1) # masses for each particle, every system has 4 particles


###############################################################################################################
## Set Parameters                                                                                            ##
###############################################################################################################


# Simulation parameters
masses = np.ones(4)  # masses for each particle, every system has 4 particles

box_size = np.sqrt(20) # simulation box size

stop_time = 10 # simulation time
dt = 0.001 # time step


epsilon = 1
sigma = 1 # distance at which the inter-particle potential is zero, in this case the distance between two particles is 1.0 when they are next to each other

###############################################################################################################
## Create lists for positions and velocities of each system                                                  ##
###############################################################################################################

print(tensor.dtype)

initial_positions = []
initial_velocities = []
for i in range(len(tensor)):
    initial_positions.append(tensor[i][0][0]) # initial positions of the molecules in time step 0 [system_index][0 for positions][timestep]
    initial_velocities.append(tensor[i][1][0]) # initial velocities of the molecules in time step 0 [system_index][1 for velocities][timestep]

initial_positions = [pos.numpy() for pos in initial_positions]
initial_velocities = [vel.numpy() for vel in initial_velocities]

###############################################################################################################
## Start of the program                                                                                      ##
###############################################################################################################

preds_tensor = torch.zeros(100, 3, 1, 4, 2)

for i in range(len(initial_positions)):
    system = i
    final_pos, final_vel = Step_until_end(initial_positions[system], initial_velocities[system], masses, epsilon, sigma, dt, stop_time, box_size)
    
    
    preds_tensor[system][0][0] = torch.tensor(final_pos, dtype=torch.float64)
    preds_tensor[system][1][0] = torch.tensor(final_vel, dtype=torch.float64)
    
preds_dict = {'qpl_trajectory': preds_tensor}

torch.save(preds_dict, r'Array/n4rho0.2T0.65t3_pred.pt')
