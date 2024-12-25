import csv
import itasca
from itasca import ball, wall
from itertools import product
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

# Setup logger at the top of the file
logger = setup_logger("batch", 'grid_search.log')

# Running experiments 101-150 (50 total)
start_comb_idx = 100
exp_num = 50

"""
start_comb_idx: Starting index for the grid search combinations (0-based)
exp_num: Number of experiments to run

1. For a fresh start:
   - start_comb_idx = 0
   - exp_num = total number of desired experiments

2. To resume from a failed experiment:
   - start_comb_idx = failed_experiment_number - 1
   - exp_num = (original_end_number - start_comb_idx + 1)

Example scenarios:
a) Running experiments 101-150 (50 total):
   start_comb_idx = 100
   exp_num = 50

b) If experiment 142 fails and you want to resume from there:
   start_comb_idx = 141  # (142 - 1)
   exp_num = 9           # (150 - 141)
"""

deterministic_mode = False
checkpoint = False

# layer_array = [
#     5,  # coal layer height
# ]
# subsurface_level = 10   # set the height of the subsurface
first_section_length = 4   # set the length of the first section
sec_interval = 3   # set the length of each section

opencut_sec = 5   # set the section where excavation starts
step_solve_time = 3  # Define step solve time

# Prevent Python state from resetting when issuing 'model new' or 'model restore'
itasca.command("python-reset-state false")

def run_simulation(params):
    try:
        itasca.command(f"model new")
        itasca.set_deterministic(deterministic_mode)
        fric, rfric, dpnr, dpsr, F0, D0 = params
        # Create result path with contact properties and date
        resu_path = f'experiments/running_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}'

        if os.path.exists(f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}'):
            logger.warning(f"The simulation in {resu_path} already exists, ending simulation.")
            return False, f"The simulation in {resu_path} already exists, ending simulation."
        
        # Create main result directory and subdirectories
        if not os.path.exists(resu_path):
            os.makedirs(resu_path)
            os.makedirs(os.path.join(resu_path, 'img'))
            os.makedirs(os.path.join(resu_path, 'sav')) 
            os.makedirs(os.path.join(resu_path, 'mat'))

        # fenceng
        # itasca.command("model restore 'yuya'")
        # sec_num = fenceng(
        #     sec_interval=sec_interval, 
        #     layer_array=layer_array,
        #     first_section_length=first_section_length,
        #     subsurface_level=subsurface_level
        # )

        # pingheng
        itasca.command("model restore 'fenceng'")

        wall_up_pos_y = wall.find('boxWallTop3').pos_y()
        wlx, wly = compute_dimensions()

        sec_num = calculate_section_number(wlx, sec_interval, first_section_length)

        # send contact properties to FISH
        itasca.fish.set('fric', fric)
        itasca.fish.set('rfric', rfric)
        itasca.fish.set('dpnr', dpnr)
        itasca.fish.set('dpsr', dpsr)
        itasca.fish.set('F0', F0)
        itasca.fish.set('D0', D0)

        run_dat_file("pingheng-linear.dat")
        if checkpoint:
            itasca.command(f"model save '{os.path.join(resu_path, 'sav', 'pingheng')}'")

        # kaiwa
        # Reset ball attributes
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")

        # get a dict ball objects of each section near the ball at the top of the model
        top_ball_pos = get_balls_max_pos(1)

        # Print warning if top_ball_pos is larger than wall_up_pos_y
        if top_ball_pos > wall_up_pos_y * 1.1:
            logger.warning(f"The model is expanding, please check the model. Current top_ball_pos: {top_ball_pos}, model height: {wall_up_pos_y}")
        if top_ball_pos < wall_up_pos_y * 0.8:
            logger.warning(f"The model is shrinking, please check the model. Current top_ball_pos: {top_ball_pos}, model height: {wall_up_pos_y}")

        rdmax = itasca.fish.get('rdmax')
        
        ball_objects_dict = {}
        for i in range(0, sec_num):
            ball_objects_dict[str(i)] = get_balls_object_in_area(str(i), top_ball_pos-rdmax*1.5, top_ball_pos)

        # delete empty list in ball_objects_dict
        empty_sections = [k for k, v in ball_objects_dict.items() if not v]
        if empty_sections:
            logger.warning(f"The following sections fetched 0 balls and will be deleted: {empty_sections}")
            ball_objects_dict = {k: v for k, v in ball_objects_dict.items() if v}
            sec_num = len(ball_objects_dict)

        y_disps_list = {}
        
        # Loop through sections
        for i in range(opencut_sec, sec_num - 3):
            if first_section_length > 0:
                excavation_pos = first_section_length + i * sec_interval
            else:
                excavation_pos = (i + 1) * sec_interval
            
            # Loop through all balls
            for ball_obj in list(ball.list()):  # Convert to list to avoid iterator issues
                if ball_obj.valid():  # Check if ball is still valid
                    if ball_obj.in_group('1', 'layer') and ball_obj.in_group(f'{i}', 'section'):
                        try:
                            ball_obj.delete()
                        except Exception as e:
                            logger.error(f"Error deleting ball {ball_obj.id()}: {str(e)}")
            
            # Save model and solve
            itasca.command(f"model solve cycle 10")
            global_timestep = itasca.timestep()
            step_interval = int(step_solve_time / global_timestep)
            if i < opencut_sec+2:
                itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-5")
            elif i < sec_num-3:
                itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-3")
            else:
                itasca.command(f"model solve ratio-average 1e-3")
            if checkpoint:
                itasca.command(f"model save '{os.path.join(resu_path, 'sav', str(i))}'")

            # get avg y disp of each section and plot the y disp vs section number
            y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(i)]) for i in range(0, sec_num)]
            y_disps_list[excavation_pos] = y_disps

            plot_y_displacement_heatmap(window_size=rdmax * 2, model_width=wlx, model_height=wly, name=excavation_pos, interpolate='nearest', resu_path=resu_path)

        # plot every y disp vs section number
        plt.figure(figsize=(10, 6))
        for excavation_pos, y_disps in y_disps_list.items():
            x_positions = [first_section_length / 2 if first_section_length > 0 else sec_interval / 2] + \
                        [first_section_length + i * sec_interval + sec_interval / 2 if first_section_length > 0 else i * sec_interval + sec_interval / 2 for i in range(0, sec_num - 2)] + \
                        [int(wlx)]
            plt.plot(x_positions, y_disps, label=f'{excavation_pos}')
        plt.xlabel('Working Face Position (m)')
        plt.xlim(0, int(wlx))
        plt.ylabel('Vertical Displacement (m)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
        plt.savefig(os.path.join(resu_path, 'img', 'surface_y_disp_vs_section.png'), dpi=400, bbox_inches='tight')
        plt.close()

        # save y_disps_list to csv
        with open(os.path.join(resu_path, 'surface_y_disp_vs_section.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header row with step numbers
            writer.writerow(['Monitoring Point'] + list(np.fromiter(y_disps_list.keys(), dtype=float)))
            # Write data rows
            for section in range(0, sec_num):
                if section == 0:
                    x_position = first_section_length / 2 if first_section_length > 0 else sec_interval / 2
                elif section == sec_num - 1:
                    x_position = int(wlx)
                else:
                    x_position = first_section_length + (section - 1) * sec_interval + sec_interval / 2 if first_section_length > 0 else (section - 1) * sec_interval + sec_interval / 2
                row = [x_position] + [y_disps_list[step][section] for step in y_disps_list]
                writer.writerow(row)

        # rename the result folder to the "experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}"
        os.rename(resu_path, f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}')
        
        return True, f"Simulation finished in {resu_path}"
        
    except Exception as e:
        logger.error(f"Error in simulation {resu_path}: {str(e)}")
        return False, f"Error in simulation {resu_path}: {str(e)}"

def main(start_comb_idx, exp_num):
    # Define parameter ranges
    # param_ranges = {
    #     'fric': np.linspace(0.0, 1.0, 4),   # 4 values from 0.0 to 1.0
    #     'rfric': np.linspace(0.0, 1.0, 4),  # 4 values from 0.0 to 1.0
    #     'dpnr': np.linspace(0.0, 1.0, 5),   # 5 values from 0.0 to 1.0
    #     'dpsr': np.linspace(0.0, 1.0, 5),   # 5 values from 0.0 to 1.0
    #     # 'F0': np.logspace(4, 6, 5)        # 5 values from 1e4 to 1e6
    #     'F0': [1e4, 1e6],
    #     'D0': [1e-6, 1]
    # }

    param_ranges = {
        'fric': np.linspace(0, 0.2, 5),   # 5 values from 0.0 to 1.0
        'rfric': [0.05,],  # 4 values from 0.0 to 1.0
        'dpnr': [0.2,],   # 5 values from 0.0 to 1.0
        'dpsr': [0.2,],   # 5 values from 0.0 to 1.0
        'F0': np.linspace(1e4, 1e5, 10),
        'D0': np.linspace(0.1, 0.5, 10)
    }
    
    # Generate all combinations
    all_combinations = list(product(
        param_ranges['fric'],
        param_ranges['rfric'],
        param_ranges['dpnr'],
        param_ranges['dpsr'],
        param_ranges['F0'],
        param_ranges['D0']
    ))
    total_combinations = len(all_combinations)
    end_idx = min(start_comb_idx + exp_num, len(all_combinations))
    param_combinations = all_combinations[start_comb_idx:end_idx]
    
    run_combinations = len(param_combinations)
    
    logger.info(f"Starting grid search at combination {start_comb_idx + 1} with {run_combinations} combinations")
    
    # Run all combinations
    start_time = time.time()
    for i, params in enumerate(param_combinations, 1):
        logger.info(f"Running combination {i + start_comb_idx}/{total_combinations}")
        logger.info(f"Parameters: fric={params[0]}, rfric={params[1]}, dpnr={params[2]}, dpsr={params[3]}, F0={params[4]}, D0={params[5]}")
        
        success, msg = run_simulation(params)
        # success, msg = True, "Test"

        # Log progress (now handled by logger)
        logger.info(f"Status: {'Success' if success else 'Failed'}")
        logger.info(f"Message: {msg}")
        
        # Calculate and display estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_sim = elapsed_time / i
        remaining_time = avg_time_per_sim * (run_combinations - i)
        logger.info(f"Estimated time remaining: {remaining_time/3600:.1f} hours")

if __name__ == "__main__":
    main(start_comb_idx, exp_num)
