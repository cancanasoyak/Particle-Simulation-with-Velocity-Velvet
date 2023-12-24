import torch
import numpy as np
import multiprocessing
from functools import partial

from Functions import Step_until_end




def worker(system, initial_positions, initial_velocities, partial_step_until_end):
    initial_pos = initial_positions[system]
    initial_vel = initial_velocities[system]
    final_pos, final_vel = partial_step_until_end(initial_pos, initial_vel)
    return system, final_pos, final_vel



if __name__ == '__main__':

    ###############################################################################################################
    ## Load the tensor from the .pt file                                                                         ##
    ###############################################################################################################

    dict = torch.load(r'Array/n4rho0.2T0.65t3.pt')
    # keys: ['qpl_trajectory', 'tau_short', 'tau_long']
    tensor = dict['qpl_trajectory']

    # tensor has the following shape and structure:
    # shape: [100,3,4,4,2]
    
    # molecule_positions = tensor[system_index][0][time][molecule_index] is a 2D array with the x and y coordinates of the molecule
    # molecule_velocities = tensor[system_index][1][time][molecule_index] is a 2D array with the x and y velocities of the molecule
    
    # create a masses vector for each particle in each system
    # masses for each particle, every system has 4 particles
    """ masses = [] #
    
    for i in range(len(tensor)):  # for each system
        masses.append(np.array([1 for j in range(len(tensor[i][0][0]))])) # I created a masses vector with ones for every molecule
    
    masses = np.array(masses) # convert the list to a numpy array """
    
    masses = [1,1,1,1]
    
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

    initial_positions = []
    initial_velocities = []
    for i in range(len(tensor)):
        initial_positions.append(tensor[i][0][0]) # initial positions of the molecules in time step 0 [system_index][0 for positions][timestep]
        initial_velocities.append(tensor[i][1][0]) # initial velocities of the molecules in time step 0 [system_index][1 for velocities][timestep]

    initial_positions = [pos.numpy() for pos in initial_positions]
    initial_velocities = [vel.numpy() for vel in initial_velocities]
    
    
    # Create a partial function with fixed arguments for multiprocessing
    partial_step_until_end = partial(
        Step_until_end,
        masses=masses,
        epsilon=epsilon,
        sigma=sigma,
        dt=dt,
        stop_time=stop_time,
        box_size=box_size
    )
    
    partial_worker = partial(worker, partial_step_until_end=partial_step_until_end)
    
    ###############################################################################################################
    ## Start of the program                                                                                      ##
    ###############################################################################################################

    # Use multiprocessing.Pool to parallelize the computation
    with multiprocessing.Pool() as pool:
        results = pool.starmap(partial_worker, [(i, initial_positions, initial_velocities) for i in range(len(initial_positions))])

    # Results has the following structure:
    # [(system_idx, final_pos, final_vel), (system_idx, final_pos, final_vel), ...]
    
    preds_tensor = torch.zeros(100, 3, 1, 4, 2)    

    for system, final_pos, final_vel in results:
        preds_tensor[system][0][0] = torch.tensor(final_pos, dtype=torch.float64)
        preds_tensor[system][1][0] = torch.tensor(final_vel, dtype=torch.float64)
        
    preds_dict = {'qpl_trajectory': preds_tensor}

    torch.save(preds_dict, r'Array/n4rho0.2T0.65t3_pred.pt')
    