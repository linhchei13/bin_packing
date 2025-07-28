import fileinput
from itertools import chain, combinations
import math
import os
import signal
from threading import Timer
import threading
import subprocess
import logging

import pandas as pd
from pysat.formula import CNF
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from pysat.solvers import Glucose3
import datetime
import pandas as pd
import os
import sys
import time
from datetime import datetime
import json
import timeit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutException(Exception): pass

# Global variables
n_items = 0
W, H = 0, 0
upper_bound = 0
rectangles = []
variables_length = 0
clauses_length = 0

optimal_bins = 0
best_bins = 0
optimal_pos = []
optimal_rot = []
bins_used = []
instance_name = ""
config_name = "BPP_INC_S_R_SB"
start = timeit.default_timer()

# Signal handler for graceful interruption (e.g., by runlim)
def handle_interrupt(signum, frame):
    global best_bins, variables_length, clauses_length, instance_name, start, optimal_bins
    logger.info(f"Received interrupt signal {signum}. Saving current best solution.")
    
    # Get the best bins found (or upper_bound if no solution found)
    current_bins = optimal_bins if optimal_bins > 0 else (best_bins if best_bins != float('inf') else upper_bound)
    logger.info(f"Best bins found before interrupt: {current_bins}")
    
    runtime = timeit.default_timer() - start
    
    # Format output similar to SPP
    print(f"c Instance: {instance_name}")
    print(f"c Config: {config_name}")
    print(f"c Variables: {variables_length}")
    print(f"c Clauses: {clauses_length}")
    print(f"c Runtime: {runtime:.2f}s")
    print(f"c Status: TIMEOUT")
    print(f"s TIMEOUT")
    print(f"o {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instance_name,
        'Variables': variables_length,
        'Clauses': clauses_length,
        'Runtime': runtime,
        'N_Bins': current_bins,
        'Status': 'TIMEOUT',
    }
    
    with open(f'result_{instance_name}_{config_name}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    sys.exit(124)  # Standard timeout exit code

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)  # Termination signal
signal.signal(signal.SIGINT, handle_interrupt)   # Keyboard interrupt (Ctrl+C)
signal.signal(signal.SIGALRM, handle_interrupt)  # Alarm signal

# Create BPP folder if it doesn't exist
os.makedirs(config_name, exist_ok=True)

def read_file_instance(instance_name):
    """Read instance file similar to SPP format"""
    possible_paths = [
        f"inputs/BENG/{instance_name}.txt",
        f"inputs/CLASS/{instance_name}.txt", 
        f"inputs/{instance_name}.txt"
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if not filepath:
        raise FileNotFoundError(f"Cannot find input file for instance {instance_name}")
    
    lines = []
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines

def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def save_checkpoint(instance, variables, clauses, bins, status="IN_PROGRESS"):
    """Save checkpoint for recovery"""
    checkpoint_data = {
        'Instance': instance,
        'Variables': variables,
        'Clauses': clauses,
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status,
    }
    
    checkpoint_file = f'checkpoint_{instance}_{config_name}.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_file}")

def load_checkpoint(instance_name):
    """Load checkpoint if exists"""
    checkpoint_file = f'checkpoint_{instance_name}_{config_name}.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def write_excel_results(results, excel_file):
    """Write results to Excel file similar to SPP"""
    try:
        # Try to load existing Excel file
        if os.path.exists(excel_file):
            df_existing = pd.read_excel(excel_file)
            df_new = pd.DataFrame([results])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame([results])
        
        # Save to Excel
        df_combined.to_excel(excel_file, index=False)
        logger.info(f"Results written to Excel: {excel_file}")
        
    except Exception as e:
        logger.error(f"Error writing to Excel: {e}")

def run_with_runlim(command, timeout, memory_limit):
    """Run command with runlim for proper resource management"""
    runlim_cmd = f"./runlim --time-limit={timeout} {command}"
    logger.info(f"Running with runlim: {runlim_cmd}")
    
    try:
        result = subprocess.run(runlim_cmd.split(), 
                              capture_output=True, 
                              text=True, 
                              timeout=timeout + 60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Timeout"  # 124 is timeout exit code

def first_fit_upper_bound(rectangles, W, H):
    """Finite First-Fit (FFF) upper bound for 2D bin packing with rotation (Berkey & Wang)."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        # For simplicity, try to place at (0, y) for all y up to H-h
        # and check for overlap with all placed rectangles
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                rect = (x, y, w, h)
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    for rect in rectangles:
        placed = False
        for bin_rects in bins:
            # Try both orientations in this bin
            for (rw, rh) in [(rect[0], rect[1]), (rect[1], rect[0])]:
                pos = fits(bin_rects, rw, rh, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], rw, rh))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Start a new bin, place at (0,0) in best orientation
            if rect[0] <= W and rect[1] <= H:
                bins.append([(0, 0, rect[0], rect[1])])
            elif rect[1] <= W and rect[0] <= H:
                bins.append([(0, 0, rect[1], rect[0])])
            else:
                # Infeasible rectangle
                return float('inf')
    return len(bins)


def BPP_incremental(W, H, lower_bound, upper_bound):
    """Incremental SAT-based BPP solver with stacking and Config 2 symmetry breaking"""
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot, bins_used
    global instance_name
    
    height = upper_bound * H
    width = W
    cnf = CNF()
    variables = {}
    id_variables = 1
    start_encoding = time.time()
    
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = id_variables  # lri,rj
                id_variables += 1
                variables[f"ud{i + 1},{j + 1}"] = id_variables  # uri,rj
                id_variables += 1
        for e in range(width):
            variables[f"px{i + 1},{e}"] = id_variables  # pxi,e
            id_variables += 1
        for f in range(height):
            variables[f"py{i + 1},{f}"] = id_variables  # pyi,f
            id_variables += 1

    # Rotated variables
    for i in range(len(rectangles)):
        variables[f"r{i + 1}"] = id_variables
        id_variables += 1
    # Height variables - ph_h means "can pack with height ≤ h"
    for h in range(lower_bound * H, upper_bound * H + 1):
        variables[f"ph_{h}"] = id_variables
        id_variables += 1
    
    # Height ordering constraints
    for h in range(lower_bound * H, upper_bound * H):
        cnf.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
    
    # Add the 2-literal axiom clauses
    for i in range(len(rectangles)):
        for e in range(width - 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        for f in range(height - 1):  # -1 because we're using f+1 in the clause
            cnf.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])
            
    # Add non-overlapping constraints
    def non_overlapping(rotated, i, j, h1, h2, v1, v2):
        # Get dimensions based on rotation
        if not rotated:
            i_width = rectangles[i][0]
            i_height = rectangles[i][1]
            j_width = rectangles[j][0]
            j_height = rectangles[j][1]
            i_rotation = variables[f"r{i + 1}"]
            j_rotation = variables[f"r{j + 1}"]
        else:
            i_width = rectangles[i][1]
            i_height = rectangles[i][0]
            j_width = rectangles[j][1]
            j_height = rectangles[j][0]
            i_rotation = -variables[f"r{i + 1}"]
            j_rotation = -variables[f"r{j + 1}"]

        # lri,j v lrj,i v udi,j v udj,i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])

        cnf.append(four_literal + [i_rotation])
        cnf.append(four_literal + [j_rotation])

        # Optimize range checks
        max_width = min(width, max(i_width, j_width))
        max_height = min(height, max(i_height, j_height))

        # Add constraints only if they're necessary
        if h1:
            for e in range(min(width, i_width)):
                cnf.append([i_rotation, -variables[f"lr{i + 1},{j + 1}"], -variables[f"px{j + 1},{e}"]])
        
        if h2:
            for e in range(min(width, j_width)):
                cnf.append([j_rotation,
                            -variables[f"lr{j + 1},{i + 1}"],
                            -variables[f"px{i + 1},{e}"]])
        # ¬udi,j ∨ ¬pyj,f
        if v1:
            for f in range(min(height, i_height)):
                cnf.append([i_rotation,
                            -variables[f"ud{i + 1},{j + 1}"],
                            -variables[f"py{j + 1},{f}"]])
        # ¬udj,i ∨ ¬pyi,f
        if v2:
            for f in range(min(height, j_height)):
                cnf.append([j_rotation,
                            -variables[f"ud{j + 1},{i + 1}"],
                            -variables[f"py{i + 1},{f}"]])

        for e in positive_range(width - i_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1:
                cnf.append([i_rotation,
                            -variables[f"lr{i + 1},{j + 1}"],
                            variables[f"px{i + 1},{e}"],
                            -variables[f"px{j + 1},{e + i_width}"]])

        for e in positive_range(width - j_width):
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2:
                cnf.append([j_rotation,
                            -variables[f"lr{j + 1},{i + 1}"],
                            variables[f"px{j + 1},{e}"],
                            -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(height - i_height):
            # ¬udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1:
                cnf.append([i_rotation,
                            -variables[f"ud{i + 1},{j + 1}"],
                            variables[f"py{i + 1},{f}"],
                            -variables[f"py{j + 1},{f + i_height}"]])
        for f in positive_range(height - j_height):
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2:
                cnf.append([j_rotation,
                            -variables[f"ud{j + 1},{i + 1}"],
                            variables[f"py{j + 1},{f}"],
                            -variables[f"py{i + 1},{f + j_height}"]])
                            
                
    max_width = max([int(rectangle[0]) for rectangle in rectangles])
    second_max_width = max([int(rectangle[0]) for rectangle in rectangles if int(rectangle[0]) != max_width])

    #Symmetry Breaking
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            #Fix the position of the largest rectangle and the second largest rectangle
            if rectangles[i][0] == max_width and rectangles[j][0] == second_max_width:
                non_overlapping(False, i, j, False, False, True, True)
                non_overlapping(True, i, j, False, False, True, True)
            # Large-rectangles horizontal
            if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > width:
                non_overlapping(False, i, j, False, False, True, True)
                non_overlapping(True, i, j, False, False, True, True)
            # Large rectangles vertical
            elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > height:
                non_overlapping(False, i, j, True, True, False, False)
                non_overlapping(True, i, j, True, True, False, False)
            # Normal rectangles
            else:
                non_overlapping(False, i, j, True, True, True, True)
                non_overlapping(True, i, j, True, True, True, True)


    # # Domain encoding to ensure every rectangle stays inside strip's boundary
    for i in range(len(rectangles)):
            if rectangles[i][0] > width: #if rectangle[i]'s width larger than strip's width, it has to be rotated
                cnf.append([variables[f"r{i + 1}"]])
            else:
                cnf.append([variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][0]}"]])
       
            if rectangles[i][1] > height:
                cnf.append([variables[f"r{i + 1}"]])
            else:
                cnf.append([variables[f"r{i + 1}"],
                            variables[f"py{i + 1},{height - rectangles[i][1]}"]])

            # Rotated
            if rectangles[i][1] > width:
                cnf.append([-variables[f"r{i + 1}"]])
            else:
                cnf.append([-variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][1]}"]])
            if rectangles[i][0] > height:
                cnf.append([-variables[f"r{i + 1}"]])
            else:
                cnf.append([-variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{height - rectangles[i][0]}"]])
    
    for k in range(1, upper_bound):
        for i in range(len(rectangles)):
            # Not rotated
            if rectangles[i][1] > H:
                cnf.append([variables[f"r{i + 1}"]])
            else:
                h = rectangles[i][1]
                cnf.append([variables[f"r{i + 1}"], variables[f"py{i + 1},{k * H - h}"],
                            -variables[f"py{i + 1},{k * H - 1}"]])
                cnf.append([variables[f"r{i + 1}"], -variables[f"py{i + 1},{k * H - h}"],
                            variables[f"py{i + 1},{k * H - 1}"]])
                
            # Rotated
            if rectangles[i][0] > H:
                cnf.append([-variables[f"r{i + 1}"]])
            else:
                cnf.append([-variables[f"r{i + 1}"], variables[f"py{i + 1},{k * H - rectangles[i][0]}"],
                            -variables[f"py{i + 1},{k * H - 1}"]])
                cnf.append([-variables[f"r{i + 1}"], -variables[f"py{i + 1},{k * H - rectangles[i][0]}"],
                            variables[f"py{i + 1},{k * H - 1}"]])

    # Height constraints - connecting ph_h with rectangle positions
    for h in range(lower_bound* H, upper_bound*H + 1):
        for i in range(len(rectangles)):
            # Normal orientation
            rect_height = rectangles[i][1]
            if h >= rect_height:
                cnf.append([-variables[f"ph_{h}"], variables[f"r{i+1}"], 
                          variables[f"py{i+1},{h - rect_height}"]])
            
            # Rotated orientation
            rotated_height = rectangles[i][0]
            if h >= rotated_height:
                cnf.append([-variables[f"ph_{h}"], -variables[f"r{i+1}"], 
                          variables[f"py{i+1},{h - rotated_height}"]])
                
    print("Encoding Time:", format(time.time() - start_encoding, ".6f"))
    variables_length = len(variables)
    clauses_length = len(cnf.clauses)
    save_checkpoint(instance_name, variables_length, clauses_length, best_bins, "IN_PROGRESS")
    
    with Glucose3(use_timer=True) as solver:
        # Add all clauses to the solver
        solver.append_formula(cnf)
        best_model = None
        positions = None
        
        # Binary search with incremental solving
        current_lb = lower_bound
        current_ub = upper_bound
        
        while current_lb <= current_ub:
            mid = (current_lb + current_ub) // 2
            print(f"Trying height: {mid*H} (lower={current_lb}, upper={current_ub})")
            
            # Set up assumptions for this iteration - test if we can pack with height ≤ mid
            assumptions = [variables[f"ph_{mid*H}"]]
            
            # Save checkpoint before solving
            save_checkpoint(instance_name, variables_length, clauses_length, 
                         best_bins if best_bins != float('inf') else upper_bound)
            
            # Solve with assumptions
            is_sat = solver.solve(assumptions=assumptions)
            
            if is_sat:
                # We found a solution with height ≤ mid
                print(f"Found solution with height ≤ {mid*H}")
                best_bins = mid
                optimal_bins = best_bins
                
                save_checkpoint(instance_name, variables_length, clauses_length, best_bins)
                
                # Save the model for solution extraction
                best_model = solver.get_model()
                
                # Extract positions from the model
                positions = [[0, 0] for _ in range(n_items)]
                rotations = [False for _ in range(n_items)]
                
                # Convert model to dictionary for faster lookup
                true_vars = set(var for var in best_model if var > 0)
                
                # Extract rotation variables
                for i in range(n_items):
                    rotations[i] = variables[f"r{i+1}"] in true_vars
                
                # Extract rotation variables
                for i in range(n_items):
                    rotations[i] = variables[f"r{i+1}"] in true_vars
                
                # Extract positions
                for i in range(n_items):
                    # Find x position (first position where px is true)
                    found_x = False
                    for e in range(W):
                        var = variables.get(f"px{i+1},{e}", None)
                        if var in true_vars:
                            if e == 0 or variables[f"px{i+1},{e-1}"] not in true_vars:
                                positions[i][0] = e
                                found_x = True
                                break
                    if not found_x:
                        print(f"WARNING: Could not determine x-position for rectangle {i}!")
                    
                    # Find y position (first position where py is true)
                    found_y = False
                    for y in range(height):
                        var = variables.get(f"py{i+1},{y}", None)
                        if var in true_vars:
                            if y == 0 or variables[f"py{i+1},{y-1}"] not in true_vars:
                                positions[i][1] = y % H # position in the bin
                                positions[i].append(y // H)  # Add bin index
                                found_y = True
                                break
                    if not found_y:
                        print(f"WARNING: Could not determine y-position for rectangle {i}!")
                
                # Save the best positions
                optimal_pos = positions
                optimal_rot = rotations
                
                # Create bins_used structure
                bins_used = [[] for _ in range(mid)]
                for i in range(n_items):
                    if len(positions[i]) > 2:
                        bin_idx = positions[i][2]
                        if 0 <= bin_idx < len(bins_used):
                            bins_used[bin_idx].append(i)
                
                # Update search range - try lower height
                current_ub = mid - 1
            
            else:
                # No solution with height ≤ mid
                print(f"No solution with height ≤ {mid*H}, trying higher")
                current_lb = mid + 1

        variables_length = len(variables)
        clauses_length = len(cnf.clauses)

        # Final validation of the solution
        if positions is None:
            return None, None
        
        print(f"Final optimal height: {optimal_bins}")
        return optimal_bins, positions
    
        

def display_solution_each_bin(W, H, rectangles, positions, rotations, bins_used):
    """Display all bins in one window with subplots"""
    
    # Use the new colormap API for compatibility
    n_bins = len(bins_used)
    ncols = min(n_bins, 4)
    nrows = (n_bins + ncols - 1) // ncols
    plt.title(f'Solution for {instance_name} - {config_name}')

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    fig.suptitle(f'Solution for {instance_name} - {config_name} - {n_bins} bins', fontsize=16)
    # Handle different subplot configurations
    if n_bins == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    
    axes = axes.flatten()
    
    for bin_idx, items_in_bin in enumerate(bins_used):
        ax = axes[bin_idx]
        ax.set_title(f'Bin {bin_idx + 1}')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        
        # Draw rectangles for items in this bin
        for item_idx in items_in_bin:
            #rotated = rotations[item_idx]
            rotated = "R" if rotations[item_idx] else ""
            if rotations[item_idx]:
                # If rotated, swap width and height
                w, h = rectangles[item_idx][1], rectangles[item_idx][0]
            else:
                w, h = rectangles[item_idx]
            # For incremental approach, positions contain bin-relative coordinates
            if len(positions[item_idx]) >= 2:
                x0, y0 = positions[item_idx][0], positions[item_idx][1]
            else:
                continue
            
            rect_patch = plt.Rectangle((x0, y0), w, h, 
                                     edgecolor='black', 
                                     facecolor="lightblue", 
                                     alpha=0.7)
            ax.add_patch(rect_patch)
            
            # Add item number in the center
            cx, cy = x0 + w/2, y0 + h/2
            
            
            ax.text(cx, cy, f'{item_idx + 1}' + rotated, 
                   ha='center', va='center', 
                    fontweight='bold', fontsize=8)
        
        # Set grid and ticks
        ax.set_xticks(range(0, W+1, max(1, W//10)))
        ax.set_yticks(range(0, H+1, max(1, H//10)))
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save plot instead of showing for better compatibility
    try:
        plt.savefig(f'{config_name}/{instance_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Solution visualization saved to {config_name}/{instance_name}.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
    

# Main
def write_excel_results(result_dict, excel_file=f'{config_name}.xlsx'):
    """Write results to Excel file similar to SPP format"""
    try:
        df = pd.DataFrame([result_dict])
        
        if os.path.exists(excel_file):
            existing_df = pd.read_excel(excel_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_excel(excel_file, index=False)
        logger.info(f"Results written to {excel_file}")
    except Exception as e:
        logger.error(f"Error writing to Excel: {e}")

def load_checkpoint(instance):
    """Load checkpoint if exists"""
    checkpoint_file = f'checkpoint_{instance}_{config_name}.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def run_with_runlim(command, timeout, memory_limit):
    """Run command with runlim for proper resource management"""
    runlim_cmd = f"./runlim -t {timeout} {command}"
    logger.info(f"Running with runlim: {runlim_cmd}")
    
    try:
        result = subprocess.run(runlim_cmd.split(), 
                              capture_output=True, 
                              text=True, 
                              timeout=timeout + 60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Timeout"  # 124 is timeout exit code


def solve_single_instance(instance_name_param, time_limit):
    """Solve a single instance with proper error handling"""
    global start, best_bins, variables_length, clauses_length, W, H, n_items
    global rectangles, optimal_bins, optimal_pos, instance_name, bins_used, upper_bound
    
    # Set the global instance_name
    instance_name = instance_name_param
    
    try:
        logger.info(f"Solving instance: {instance_name}")
        start = timeit.default_timer()

        # Read input
        input_data = read_file_instance(instance_name)
        n_items = int(input_data[0])
        W, H = map(int, input_data[1].split())
        rectangles = [[int(val) for val in i.split()] for i in input_data[2:]]
        
        logger.info(f"Instance {instance_name}: {n_items} items, bin size {W}x{H}")
        print(f"Number of items: {n_items}, Width: {W}, Height: {H}")
        
        # Calculate bounds
        total_area = sum([w * h for w, h in rectangles])
        lower_bound = math.ceil(total_area / (W * H))
        print(f"Lower bound: {lower_bound}")
        upper_bound = first_fit_upper_bound(rectangles, W, H)
        best_bins = upper_bound
        
        # Solve using incremental approach
        result = BPP_incremental(W, H, lower_bound, upper_bound)
        optimal_bins, optimal_pos = result if result[0] is not None else (upper_bound, [])
        
        runtime = timeit.default_timer() - start
        
        # Process result
        if optimal_bins > 0 and optimal_pos:
            status = "OPTIMAL"
            n_bins = optimal_bins
            logger.info(f"Optimal solution found: {n_bins} bins")
        else:
            n_bins = best_bins if best_bins != float('inf') else upper_bound
            status = "UNSAT"
            logger.warning(f"No solution found. Tried up to {upper_bound} bins")
        
        # Format output similar to SPP
        print(f"c Instance: {instance_name}")
        print(f"c Config: {config_name}")
        print(f"c Variables: {variables_length}")
        print(f"c Clauses: {clauses_length}")
        print(f"c Runtime: {runtime:.2f}s")
        print(f"c Status: {status}")
        print(f"s {status}")
        print(f"o {n_bins}")
        
        # Save results
        result_dict = {
            'Instance': instance_name,
            'Variables': variables_length,
            'Clauses': clauses_length,
            'Runtime': runtime,
            'N_Bins': n_bins,
            'Status': status,
        }
        
        # Save JSON
        json_file = f'result_{instance_name}_{config_name}.json'
        with open(json_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"Results saved to {json_file}")
        
        excel_file = f'{config_name}.xlsx'
        # Only write to Excel if this instance is not already present
        write_to_excel = True
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
                if 'Instance' in df.columns and result_dict['Instance'] in df['Instance'].values:
                    logger.warning(f"Instance {instance_name} already exists in {excel_file}. Skipping write.")
                    write_to_excel = False
            except Exception as e:
                logger.error(f"Error reading {excel_file}: {e}")
        if write_to_excel:
            logger.info(f"Writing results to Excel: {excel_file}")
            write_excel_results(result_dict, excel_file)
        # Display solution if found
        if optimal_pos and bins_used:
            display_solution_each_bin(W, H, rectangles, optimal_pos, optimal_rot, bins_used)
        else:
            logger.warning("No valid solution found to display")

        # Clean up temporary files
        for file in [f'result_{instance_name}_{config_name}.json', f'checkpoint_{instance_name}_{config_name}.json']:
            if os.path.exists(file):
                os.remove(file)
        return result_dict
        
    except Exception as e:
        logger.error(f"Error solving {instance_name}: {e}")
        runtime = timeit.default_timer() - start if 'start' in globals() else 0
        
        error_result = {
            'Instance': instance_name,
            'Variables': 0,
            'Clauses': 0,
            'Runtime': runtime,
            'N_Bins': 0,
            'Status': 'ERROR',
            'Error': str(e),
        }
        
        write_excel_results(error_result)
        return error_result

def controller_mode():
    """Controller mode - batch process multiple instances"""
    logger.info("Starting BPP controller mode")

    # Load instance list from file if exists
    instances_to_process = [# BENG instances (10 instances)
    "BENG01", "BENG02", "BENG03", "BENG04", "BENG05",
    "BENG06", "BENG07", "BENG08", "BENG09", "BENG10",
    
    # CLASS instances (500 instances)
    # CL_1_20_x (10 instances)
    "CL_1_20_1", "CL_1_20_2", "CL_1_20_3", "CL_1_20_4", "CL_1_20_5",
    "CL_1_20_6", "CL_1_20_7", "CL_1_20_8", "CL_1_20_9", "CL_1_20_10",
    
    # CL_1_40_x (10 instances)
    "CL_1_40_1", "CL_1_40_2", "CL_1_40_3", "CL_1_40_4", "CL_1_40_5",
    "CL_1_40_6", "CL_1_40_7", "CL_1_40_8", "CL_1_40_9", "CL_1_40_10",
    
    # CL_1_60_x (10 instances)
    "CL_1_60_1", "CL_1_60_2", "CL_1_60_3", "CL_1_60_4", "CL_1_60_5",
    "CL_1_60_6", "CL_1_60_7", "CL_1_60_8", "CL_1_60_9", "CL_1_60_10",
    
    # CL_1_80_x (10 instances)
    "CL_1_80_1", "CL_1_80_2", "CL_1_80_3", "CL_1_80_4", "CL_1_80_5",
    "CL_1_80_6", "CL_1_80_7", "CL_1_80_8", "CL_1_80_9", "CL_1_80_10",
    
    # CL_1_100_x (10 instances)
    "CL_1_100_1", "CL_1_100_2", "CL_1_100_3", "CL_1_100_4", "CL_1_100_5",
    "CL_1_100_6", "CL_1_100_7", "CL_1_100_8", "CL_1_100_9", "CL_1_100_10",
    
    # CL_2_20_x (10 instances)
    "CL_2_20_1", "CL_2_20_2", "CL_2_20_3", "CL_2_20_4", "CL_2_20_5",
    "CL_2_20_6", "CL_2_20_7", "CL_2_20_8", "CL_2_20_9", "CL_2_20_10",
    
    # CL_2_40_x (10 instances)
    "CL_2_40_1", "CL_2_40_2", "CL_2_40_3", "CL_2_40_4", "CL_2_40_5",
    "CL_2_40_6", "CL_2_40_7", "CL_2_40_8", "CL_2_40_9", "CL_2_40_10",
    
    # CL_2_60_x (10 instances)
    "CL_2_60_1", "CL_2_60_2", "CL_2_60_3", "CL_2_60_4", "CL_2_60_5",
    "CL_2_60_6", "CL_2_60_7", "CL_2_60_8", "CL_2_60_9", "CL_2_60_10",
    
    # CL_2_80_x (10 instances)
    "CL_2_80_1", "CL_2_80_2", "CL_2_80_3", "CL_2_80_4", "CL_2_80_5",
    "CL_2_80_6", "CL_2_80_7", "CL_2_80_8", "CL_2_80_9", "CL_2_80_10",
    
    # CL_2_100_x (10 instances)
    "CL_2_100_1", "CL_2_100_2", "CL_2_100_3", "CL_2_100_4", "CL_2_100_5",
    "CL_2_100_6", "CL_2_100_7", "CL_2_100_8", "CL_2_100_9", "CL_2_100_10",
    
    # CL_3_20_x (10 instances)
    "CL_3_20_1", "CL_3_20_2", "CL_3_20_3", "CL_3_20_4", "CL_3_20_5",
    "CL_3_20_6", "CL_3_20_7", "CL_3_20_8", "CL_3_20_9", "CL_3_20_10",
    
    # CL_3_40_x (10 instances)
    "CL_3_40_1", "CL_3_40_2", "CL_3_40_3", "CL_3_40_4", "CL_3_40_5",
    "CL_3_40_6", "CL_3_40_7", "CL_3_40_8", "CL_3_40_9", "CL_3_40_10",
    
    # CL_3_60_x (10 instances)
    "CL_3_60_1", "CL_3_60_2", "CL_3_60_3", "CL_3_60_4", "CL_3_60_5",
    "CL_3_60_6", "CL_3_60_7", "CL_3_60_8", "CL_3_60_9", "CL_3_60_10",
    
    # CL_3_80_x (10 instances)
    "CL_3_80_1", "CL_3_80_2", "CL_3_80_3", "CL_3_80_4", "CL_3_80_5",
    "CL_3_80_6", "CL_3_80_7", "CL_3_80_8", "CL_3_80_9", "CL_3_80_10",
    
    # CL_3_100_x (10 instances)
    "CL_3_100_1", "CL_3_100_2", "CL_3_100_3", "CL_3_100_4", "CL_3_100_5",
    "CL_3_100_6", "CL_3_100_7", "CL_3_100_8", "CL_3_100_9", "CL_3_100_10",
    
    # CL_4_20_x (10 instances)
    "CL_4_20_1", "CL_4_20_2", "CL_4_20_3", "CL_4_20_4", "CL_4_20_5",
    "CL_4_20_6", "CL_4_20_7", "CL_4_20_8", "CL_4_20_9", "CL_4_20_10",
    
    # CL_4_40_x (10 instances)
    "CL_4_40_1", "CL_4_40_2", "CL_4_40_3", "CL_4_40_4", "CL_4_40_5",
    "CL_4_40_6", "CL_4_40_7", "CL_4_40_8", "CL_4_40_9", "CL_4_40_10",
    
    # CL_4_60_x (10 instances)
    "CL_4_60_1", "CL_4_60_2", "CL_4_60_3", "CL_4_60_4", "CL_4_60_5",
    "CL_4_60_6", "CL_4_60_7", "CL_4_60_8", "CL_4_60_9", "CL_4_60_10",
    
    # CL_4_80_x (10 instances)
    "CL_4_80_1", "CL_4_80_2", "CL_4_80_3", "CL_4_80_4", "CL_4_80_5",
    "CL_4_80_6", "CL_4_80_7", "CL_4_80_8", "CL_4_80_9", "CL_4_80_10",
    
    # CL_4_100_x (10 instances)
    "CL_4_100_1", "CL_4_100_2", "CL_4_100_3", "CL_4_100_4", "CL_4_100_5",
    "CL_4_100_6", "CL_4_100_7", "CL_4_100_8", "CL_4_100_9", "CL_4_100_10",
    
    # CL_5_20_x (10 instances)
    "CL_5_20_1", "CL_5_20_2", "CL_5_20_3", "CL_5_20_4", "CL_5_20_5",
    "CL_5_20_6", "CL_5_20_7", "CL_5_20_8", "CL_5_20_9", "CL_5_20_10",
    
    # CL_5_40_x (10 instances)
    "CL_5_40_1", "CL_5_40_2", "CL_5_40_3", "CL_5_40_4", "CL_5_40_5",
    "CL_5_40_6", "CL_5_40_7", "CL_5_40_8", "CL_5_40_9", "CL_5_40_10",
    
    # CL_5_60_x (10 instances)
    "CL_5_60_1", "CL_5_60_2", "CL_5_60_3", "CL_5_60_4", "CL_5_60_5",
    "CL_5_60_6", "CL_5_60_7", "CL_5_60_8", "CL_5_60_9", "CL_5_60_10",
    
    # CL_5_80_x (10 instances)
    "CL_5_80_1", "CL_5_80_2", "CL_5_80_3", "CL_5_80_4", "CL_5_80_5",
    "CL_5_80_6", "CL_5_80_7", "CL_5_80_8", "CL_5_80_9", "CL_5_80_10",
    
    # CL_5_100_x (10 instances)
    "CL_5_100_1", "CL_5_100_2", "CL_5_100_3", "CL_5_100_4", "CL_5_100_5",
    "CL_5_100_6", "CL_5_100_7", "CL_5_100_8", "CL_5_100_9", "CL_5_100_10",
    
    # CL_6_20_x (10 instances)
    "CL_6_20_1", "CL_6_20_2", "CL_6_20_3", "CL_6_20_4", "CL_6_20_5",
    "CL_6_20_6", "CL_6_20_7", "CL_6_20_8", "CL_6_20_9", "CL_6_20_10",
    
    # CL_6_40_x (10 instances)
    "CL_6_40_1", "CL_6_40_2", "CL_6_40_3", "CL_6_40_4", "CL_6_40_5",
    "CL_6_40_6", "CL_6_40_7", "CL_6_40_8", "CL_6_40_9", "CL_6_40_10",
    
    # CL_6_60_x (10 instances)
    "CL_6_60_1", "CL_6_60_2", "CL_6_60_3", "CL_6_60_4", "CL_6_60_5",
    "CL_6_60_6", "CL_6_60_7", "CL_6_60_8", "CL_6_60_9", "CL_6_60_10",
    
    # CL_6_80_x (10 instances)
    "CL_6_80_1", "CL_6_80_2", "CL_6_80_3", "CL_6_80_4", "CL_6_80_5",
    "CL_6_80_6", "CL_6_80_7", "CL_6_80_8", "CL_6_80_9", "CL_6_80_10",
    
    # CL_6_100_x (10 instances)
    "CL_6_100_1", "CL_6_100_2", "CL_6_100_3", "CL_6_100_4", "CL_6_100_5",
    "CL_6_100_6", "CL_6_100_7", "CL_6_100_8", "CL_6_100_9", "CL_6_100_10",
    
    # CL_7_20_x (10 instances)
    "CL_7_20_1", "CL_7_20_2", "CL_7_20_3", "CL_7_20_4", "CL_7_20_5",
    "CL_7_20_6", "CL_7_20_7", "CL_7_20_8", "CL_7_20_9", "CL_7_20_10",
    
    # CL_7_40_x (10 instances)
    "CL_7_40_1", "CL_7_40_2", "CL_7_40_3", "CL_7_40_4", "CL_7_40_5",
    "CL_7_40_6", "CL_7_40_7", "CL_7_40_8", "CL_7_40_9", "CL_7_40_10",
    
    # CL_7_60_x (10 instances)
    "CL_7_60_1", "CL_7_60_2", "CL_7_60_3", "CL_7_60_4", "CL_7_60_5",
    "CL_7_60_6", "CL_7_60_7", "CL_7_60_8", "CL_7_60_9", "CL_7_60_10",
    
    # CL_7_80_x (10 instances)
    "CL_7_80_1", "CL_7_80_2", "CL_7_80_3", "CL_7_80_4", "CL_7_80_5",
    "CL_7_80_6", "CL_7_80_7", "CL_7_80_8", "CL_7_80_9", "CL_7_80_10",
    
    # CL_7_100_x (10 instances)
    "CL_7_100_1", "CL_7_100_2", "CL_7_100_3", "CL_7_100_4", "CL_7_100_5",
    "CL_7_100_6", "CL_7_100_7", "CL_7_100_8", "CL_7_100_9", "CL_7_100_10",
    
    # CL_8_20_x (10 instances)
    "CL_8_20_1", "CL_8_20_2", "CL_8_20_3", "CL_8_20_4", "CL_8_20_5",
    "CL_8_20_6", "CL_8_20_7", "CL_8_20_8", "CL_8_20_9", "CL_8_20_10",
    
    # CL_8_40_x (10 instances)
    "CL_8_40_1", "CL_8_40_2", "CL_8_40_3", "CL_8_40_4", "CL_8_40_5",
    "CL_8_40_6", "CL_8_40_7", "CL_8_40_8", "CL_8_40_9", "CL_8_40_10",
    
    # CL_8_60_x (10 instances)
    "CL_8_60_1", "CL_8_60_2", "CL_8_60_3", "CL_8_60_4", "CL_8_60_5",
    "CL_8_60_6", "CL_8_60_7", "CL_8_60_8", "CL_8_60_9", "CL_8_60_10",
    
    # CL_8_80_x (10 instances)
    "CL_8_80_1", "CL_8_80_2", "CL_8_80_3", "CL_8_80_4", "CL_8_80_5",
    "CL_8_80_6", "CL_8_80_7", "CL_8_80_8", "CL_8_80_9", "CL_8_80_10",
    
    # CL_8_100_x (10 instances)
    "CL_8_100_1", "CL_8_100_2", "CL_8_100_3", "CL_8_100_4", "CL_8_100_5",
    "CL_8_100_6", "CL_8_100_7", "CL_8_100_8", "CL_8_100_9", "CL_8_100_10",
    
    # CL_9_20_x (10 instances)
    "CL_9_20_1", "CL_9_20_2", "CL_9_20_3", "CL_9_20_4", "CL_9_20_5",
    "CL_9_20_6", "CL_9_20_7", "CL_9_20_8", "CL_9_20_9", "CL_9_20_10",
    
    # CL_9_40_x (10 instances)
    "CL_9_40_1", "CL_9_40_2", "CL_9_40_3", "CL_9_40_4", "CL_9_40_5",
    "CL_9_40_6", "CL_9_40_7", "CL_9_40_8", "CL_9_40_9", "CL_9_40_10",
    
    # CL_9_60_x (10 instances)
    "CL_9_60_1", "CL_9_60_2", "CL_9_60_3", "CL_9_60_4", "CL_9_60_5",
    "CL_9_60_6", "CL_9_60_7", "CL_9_60_8", "CL_9_60_9", "CL_9_60_10",
    
    # CL_9_80_x (10 instances)
    "CL_9_80_1", "CL_9_80_2", "CL_9_80_3", "CL_9_80_4", "CL_9_80_5",
    "CL_9_80_6", "CL_9_80_7", "CL_9_80_8", "CL_9_80_9", "CL_9_80_10",
    
    # CL_9_100_x (10 instances)
    "CL_9_100_1", "CL_9_100_2", "CL_9_100_3", "CL_9_100_4", "CL_9_100_5",
    "CL_9_100_6", "CL_9_100_7", "CL_9_100_8", "CL_9_100_9", "CL_9_100_10",
    
    # CL_10_20_x (10 instances)
    "CL_10_20_1", "CL_10_20_2", "CL_10_20_3", "CL_10_20_4", "CL_10_20_5",
    "CL_10_20_6", "CL_10_20_7", "CL_10_20_8", "CL_10_20_9", "CL_10_20_10",
    
    # CL_10_40_x (10 instances)
    "CL_10_40_1", "CL_10_40_2", "CL_10_40_3", "CL_10_40_4", "CL_10_40_5",
    "CL_10_40_6", "CL_10_40_7", "CL_10_40_8", "CL_10_40_9", "CL_10_40_10",
    
    # CL_10_60_x (10 instances)
    "CL_10_60_1", "CL_10_60_2", "CL_10_60_3", "CL_10_60_4", "CL_10_60_5",
    "CL_10_60_6", "CL_10_60_7", "CL_10_60_8", "CL_10_60_9", "CL_10_60_10",
    
    # CL_10_80_x (10 instances)
    "CL_10_80_1", "CL_10_80_2", "CL_10_80_3", "CL_10_80_4", "CL_10_80_5",
    "CL_10_80_6", "CL_10_80_7", "CL_10_80_8", "CL_10_80_9", "CL_10_80_10",
    
    # CL_10_100_x (10 instances)
    "CL_10_100_1", "CL_10_100_2", "CL_10_100_3", "CL_10_100_4", "CL_10_100_5",
    "CL_10_100_6", "CL_10_100_7", "CL_10_100_8", "CL_10_100_9", "CL_10_100_10"]
    # Read existing Excel file to check completed instances
    excel_file = f'{config_name}.xlsx'
    completed_instances = []
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
        except Exception as e:
            logger.error(f"Error reading existing Excel file: {e}")
    
    # Set timeout in seconds
    TIMEOUT = 900 

    for instance_name in instances_to_process:
        # Skip if already completed
        if instance_name in completed_instances:
            instance_name = instance_name.strip()
            logger.info(f"Skipping {instance_name} - already completed")
            continue
            
        logger.info(f"Processing {instance_name}...")
        
        # Clean up any previous result files
        
        try:
            # Run with runlim if available
            if os.path.exists('./runlim'):
                script_path = os.path.abspath(__file__)
                command = f"python {script_path} {instance_name}"
                exit_code, stdout, stderr = run_with_runlim(command, timeout=TIMEOUT, memory_limit=8192)
                
                if exit_code == 0:
                    logger.info(f"Completed {instance_name} successfully")
                elif exit_code == 124:
                    logger.warning(f"Timeout for {instance_name}")
                else:
                    logger.error(f"Error for {instance_name}: exit code {exit_code}")
                    logger.error(f"stdout: {stdout}")
                    logger.error(f"stderr: {stderr}")
                    
                # Wait a moment to ensure file is written
                time.sleep(1)
                
                # Check results
                result_file = f'result_{instance_name}_{config_name}.json'
                checkpoint_file = f'checkpoint_{instance_name}_{config_name}.json'
                
                result = None
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                elif os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        result = json.load(f)
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                
                if result:
                    logger.info(f"Instance {instance_name} - Status: {result['Status']}")
                    if 'N_Bins' in result:
                        logger.info(f"N_Bins: {result['N_Bins']}, Runtime: {result['Runtime']}")
                    
                    # Write to Excel if it's a timeout result
                    if result['Status'] == 'TIMEOUT':
                        result['Runtime'] = 'TIMEOUT'
                        write_excel_results(result, excel_file)
                        
            else:
                # Run directly without runlim
                result = solve_single_instance(instance_name, time_limit=TIMEOUT)
                if result:
                    logger.info(f"Completed {instance_name} directly")
                    
        except Exception as e:
            logger.error(f"Failed to process {instance_name}: {e}")
            continue
        
        # Clean up result files to avoid confusion
        for file in [f'result_{instance_name}_{config_name}.json', f'checkpoint_{instance_name}_{config_name}.json']:
            if os.path.exists(file):
                os.remove(file)
    
    logger.info(f"All instances completed. Results saved to {excel_file}")

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(config_name, exist_ok=True)
    
    if len(sys.argv) == 1:
        # Controller mode - batch processing
        controller_mode()
    elif len(sys.argv) == 2:
        # Single instance mode
        instance_name = sys.argv[1]
        solve_single_instance(instance_name, time_limit=900)
    else:
        print(f"Usage:")
        print(f"  Controller mode: python {sys.argv[0]}")
        print(f"  Single instance: python {sys.argv[0]} <instance_name>")
        print(f"Examples:")
        print(f"  python {sys.argv[0]} BENG01")
        print(f"  python {sys.argv[0]} CL_1_20_1")
        sys.exit(1)

