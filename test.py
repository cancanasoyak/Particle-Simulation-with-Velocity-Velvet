import torch
import numpy as np
import multiprocessing
from functools import partial

from tools.Functions import Step_until_end




def worker(system, initial_positions, initial_velocities, partial_step_until_end):
    initial_pos = initial_positions[system]
    initial_vel = initial_velocities[system]
    final_pos, final_vel = partial_step_until_end(initial_pos, initial_vel)
    return system, final_pos, final_vel



if __name__ == '__main__':

    ###############################################################################################################
    ## Load the tensor from the .pt file                                                                         ##
    ###############################################################################################################

    dict = torch.load(r'Array\n4rho0.2T0.65t3.pt')
    # keys: ['qpl_trajectory', 'tau_short', 'tau_long']
    tensor = dict['qpl_trajectory']


    # molecule_positions = tensor[system_index][0][time][molecule_index]
    # molecule_velocities = tensor[system_index][1][time][molecule_index]




    ###############################################################################################################
    ## Set Parameters                                                                                            ##
    ###############################################################################################################


    # Simulation parameters
    masses = np.ones(4)  # masses for each particle, every system has 4 particles

    box_size = np.sqrt(20) # simulation box size

    stop_time = 10.0 # simulation time
    dt = 0.001 # time step


    epsilon = 1
    sigma = 1 # distance at which the inter-particle potential is zero, in this case the distance between two particles is 1.0 when they are next to each other

    ###############################################################################################################
    ## Set Initial Conditions                                                                                    ##
    ###############################################################################################################


    initial_positions = []
    initial_velocities = []
    for i in range(len(tensor)):
        initial_positions.append(tensor[i][0][0]) # initial positions of the molecules in time step 0 [system_index][0 for positions][timestep]
        initial_velocities.append(tensor[i][1][0]) # initial velocities of the molecules in time step 0 [system_index][1 for velocities][timestep]

    initial_positions = [np.array(pos) for pos in initial_positions]
    initial_velocities = [np.array(vel) for vel in initial_velocities]

    
    # Create a partial function with fixed arguments
    partial_step_until_end = partial(
        Step_until_end,
        masses=masses,
        epsilon=epsilon,
        sigma=sigma,
        dt=dt,
        stop_time=0.003,
        box_size=box_size
    )
    
    partial_worker = partial(worker, partial_step_until_end=partial_step_until_end)
    
    ###############################################################################################################
    ## Start of the program                                                                                      ##
    ###############################################################################################################

    # Use multiprocessing.Pool to parallelize the computation
    with multiprocessing.Pool() as pool:
        results = pool.starmap(partial_worker, [(i, initial_positions, initial_velocities) for i in range(len(initial_positions))])



    for system, final_pos, final_vel in results:
        print("System: ", system)
        diff_pos = final_pos - tensor[system][0][3].numpy()
        diff_vel = final_vel - tensor[system][1][3].numpy()
        print("Final Positions: ", diff_pos)
        print("Final Velocities: ", diff_vel)
    