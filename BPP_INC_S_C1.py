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
bins_used = []
instance_name = ""
config_name = "BPP_INC_S_C1"
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
# signal.signal(signal.SIGALRM, handle_interrupt)  # Alarm signal

# Create BPP folder if it doesn't exist
os.makedirs(config_name, exist_ok=True)

start = timeit.default_timer()
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
    """Finite First-Fit (FFF) upper bound for 2D bin packing without rotation."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
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
        w, h = rect[0], rect[1]
        
        # Check if rectangle fits in bin at all
        if w > W or h > H:
            return float('inf')  # Infeasible rectangle
        
        # Try to place in existing bins
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break
        
        if not placed:
            # Start a new bin, place at (0,0)
            bins.append([(0, 0, w, h)])
    
    return len(bins)

def BPP_incremental(W, H, lower_bound, upper_bound):
    """Incremental SAT-based BPP solver with stacking and Config 2 symmetry breaking"""
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, bins_used
    global instance_name
    
    height = upper_bound * H
    width = W
    clauses = []
    variables = {}
    id_variables = 1
    start_encoding = time.time()
    
    # Create variables for left-right and up-down ordering
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = id_variables  # lri,rj
                id_variables += 1
                variables[f"ud{i + 1},{j + 1}"] = id_variables  # uri,rj
                id_variables += 1
        
        # Position variables with proper domain constraints
        for e in positive_range(width - rectangles[i][0] + 1):
            variables[f"px{i + 1},{e}"] = id_variables  # pxi,e
            id_variables += 1
        for f in positive_range(height - rectangles[i][1] + 1):
            variables[f"py{i + 1},{f}"] = id_variables  # pyi,f
            id_variables += 1

    # Height variables - ph_h means "can pack with height ≤ h"
    for h in range(lower_bound * H, upper_bound * H + 1):
        variables[f"ph_{h}"] = id_variables
        id_variables += 1
    
    # Height ordering constraints
    for h in range(lower_bound * H, upper_bound * H):
        clauses.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
    
    # Add the 2-literal axiom clauses (order constraint)
    for i in range(len(rectangles)):
        # Horizontal ordering
        for e in range(width - rectangles[i][0]):
            clauses.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        # Vertical ordering
        for f in range(height - rectangles[i][1]):
            clauses.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])

    # Add non-overlapping constraints with Config 2 symmetry breaking
    def non_overlapping(i, j, h1, h2, v1, v2):
        i_width = rectangles[i][0]
        i_height = rectangles[i][1]
        j_width = rectangles[j][0]
        j_height = rectangles[j][1]

        # Four-literal disjunction for non-overlapping
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
        clauses.append(four_literal)

        # Channeling constraints for left-right ordering
        if h1:
            for e in range(i_width):
                if f"px{j + 1},{e}" in variables:
                    clauses.append([-variables[f"lr{i + 1},{j + 1}"],
                                -variables[f"px{j + 1},{e}"]])
        if h2:
            for e in range(j_width):
                if f"px{i + 1},{e}" in variables:
                    clauses.append([-variables[f"lr{j + 1},{i + 1}"],
                                -variables[f"px{i + 1},{e}"]])
        
        # Channeling constraints for up-down ordering
        if v1:
            for f in range(i_height):
                if f"py{j + 1},{f}" in variables:
                    clauses.append([-variables[f"ud{i + 1},{j + 1}"],
                                -variables[f"py{j + 1},{f}"]])
        if v2:
            for f in range(j_height):
                if f"py{i + 1},{f}" in variables:
                    clauses.append([-variables[f"ud{j + 1},{i + 1}"],
                                -variables[f"py{i + 1},{f}"]])

        # No-overlap constraints for horizontal direction
        for e in positive_range(width - i_width):
            if h1 and f"px{j + 1},{e + i_width}" in variables:
                clauses.append([-variables[f"lr{i + 1},{j + 1}"],
                            variables[f"px{i + 1},{e}"],
                            -variables[f"px{j + 1},{e + i_width}"]])
            if h2 and f"px{i + 1},{e + j_width}" in variables:
                clauses.append([-variables[f"lr{j + 1},{i + 1}"],
                            variables[f"px{j + 1},{e}"],
                            -variables[f"px{i + 1},{e + j_width}"]])

        # No-overlap constraints for vertical direction
        for f in positive_range(height - i_height):
            if v1 and f"py{j + 1},{f + i_height}" in variables:
                clauses.append([-variables[f"ud{i + 1},{j + 1}"],
                            variables[f"py{i + 1},{f}"],
                            -variables[f"py{j + 1},{f + i_height}"]])
            if v2 and f"py{i + 1},{f + j_height}" in variables:
                clauses.append([-variables[f"ud{j + 1},{i + 1}"],
                            variables[f"py{j + 1},{f}"],
                            -variables[f"py{i + 1},{f + j_height}"]])

    # Find max dimensions for symmetry breaking
    max_height = max([int(rectangle[1]) for rectangle in rectangles])
    max_width = max([int(rectangle[0]) for rectangle in rectangles])

    # Apply Config 2 symmetry breaking
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            # Large-rectangles horizontal
            if rectangles[i][0] + rectangles[j][0] > width:
                non_overlapping(i, j, False, False, True, True)
            # Large-rectangles vertical
            elif rectangles[i][1] + rectangles[j][1] > height:
                non_overlapping(i, j, True, True, False, False)
            # Same-sized rectangles
            elif rectangles[i] == rectangles[j]:
                non_overlapping(i, j, True, False, True, True)
            # Domain 
            elif rectangles[i][0] == max_width and rectangles[j][0] > (width -max_width) /2:
                non_overlapping(i, j, False, True, True, True)
            elif rectangles[i][1] == max_height and rectangles[j][1] > (height - max_height) / 2:
                non_overlapping(i, j, True, True, False, True)
            else:
                non_overlapping(i, j, True, True, True, True)

    # Domain constraints - rectangles must fit within strip boundaries
    for i in range(len(rectangles)):
        if rectangles[i][0] == max_width:
            clauses.append([variables[f"px{i + 1},{math.ceil((width - rectangles[i][0])/2)}"]])
        else:
            if f"px{i + 1},{width - rectangles[i][0]}" in variables:
                clauses.append([variables[f"px{i + 1},{width - rectangles[i][0]}"]])
        if f"py{i + 1},{height - rectangles[i][1]}" in variables:
            clauses.append([variables[f"py{i + 1},{height - rectangles[i][1]}"]])
    
    # Bin separation constraints
    for k in range(1, upper_bound):
        for i in range(len(rectangles)):
            # Constraint: pyi,k*H-1 <-> pyi,k*H-rectangles[i][1]
            if f"py{i + 1},{k * H - 1}" in variables and f"py{i + 1},{k * H - rectangles[i][1]}" in variables:
                clauses.append([-variables[f"py{i + 1},{k * H - 1}"], 
                            variables[f"py{i + 1},{k * H - rectangles[i][1]}"]])
                clauses.append([variables[f"py{i + 1},{k * H - 1}"], 
                            -variables[f"py{i + 1},{k * H - rectangles[i][1]}"]])

    # Height limit constraints
    for h in range(lower_bound * H, upper_bound * H + 1):
        for i in range(len(rectangles)):
            rect_height = rectangles[i][1]
            if h >= rect_height and f"py{i+1},{h - rect_height}" in variables:
                clauses.append([-variables[f"ph_{h}"], 
                          variables[f"py{i+1},{h - rect_height}"]])
    
    print("Encoding Time:", format(time.time() - start_encoding, ".6f"))
    variables_length = len(variables)
    clauses_length = len(clauses)
    save_checkpoint(instance_name, variables_length, clauses_length, best_bins, "IN_PROGRESS")
    
    with Glucose3(use_timer=True) as solver:
        # Add all clauses to the solver
        for clause in clauses:
            solver.add_clause(clause)
        
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
                
                # Convert model to dictionary for faster lookup
                true_vars = set(var for var in best_model if var > 0)
                
                # Extract positions using the ordering constraints
                for i in range(n_items):
                    # Find x position (first position where px is true)
                    found_x = False
                    for e in range(width - rectangles[i][0] + 1):
                        var = variables.get(f"px{i+1},{e}", None)
                        if var and var in true_vars:
                            if e == 0 or variables.get(f"px{i+1},{e-1}", 0) not in true_vars:
                                positions[i][0] = e
                                found_x = True
                                break
                    if not found_x:
                        print(f"WARNING: Could not determine x-position for rectangle {i}!")
                    
                    # Find y position (first position where py is true)
                    found_y = False
                    for y in range(height - rectangles[i][1] + 1):
                        var = variables.get(f"py{i+1},{y}", None)
                        if var and var in true_vars:
                            if y == 0 or variables.get(f"py{i+1},{y-1}", 0) not in true_vars:
                                positions[i][1] = y % H  # position in the bin
                                positions[i].append(y // H)  # Add bin index
                                found_y = True
                                break
                    if not found_y:
                        print(f"WARNING: Could not determine y-position for rectangle {i}!")
                
                # Save the best positions
                optimal_pos = positions
                
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
        clauses_length = len(clauses)

        # Final validation of the solution
        if positions is None:
            return None, None
        
        print(f"Final optimal height: {optimal_bins}")
        return optimal_bins, positions
    
        

def display_solution_each_bin(W, H, rectangles, positions, bins_used):
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
            
            
            ax.text(cx, cy, f'{item_idx + 1}', 
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
        
        print(f"Best positions: {optimal_pos}")
        print(f"Bins used: {bins_used}")
        print(f"Best bins used: {optimal_bins}")
        
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
        # Always overwrite result for this instance (update if exists, append if not)
        if os.path.exists(excel_file):
            try:
                existing_df = pd.read_excel(excel_file)
                if 'Instance' in existing_df.columns:
                    mask = existing_df['Instance'] == instance_name
                    if mask.any():
                        for key, value in result_dict.items():
                            existing_df.loc[mask, key] = value
                    else:
                        result_df = pd.DataFrame([result_dict])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                else:
                    result_df = pd.DataFrame([result_dict])
                    existing_df = pd.concat([existing_df, result_df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error updating Excel: {e}")
                existing_df = pd.DataFrame([result_dict])
        else:
            existing_df = pd.DataFrame([result_dict])
        existing_df.to_excel(excel_file, index=False)
        logger.info(f"Results written to {excel_file}")
        # Display solution if found
        if optimal_pos and bins_used:
            display_solution_each_bin(W, H, rectangles, optimal_pos, bins_used)
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
    
    # Default instances to process
    default_instances = [
        "BENG01", "BENG02", "BENG03", "BENG04", "BENG05",
        "BENG06", "BENG07", "BENG08", "BENG09", "BENG10",
        # "CL_1_20_1", "CL_1_20_2", "CL_1_20_3", "CL_1_20_4", "CL_1_20_5"
    ]
    
    # Load instance list from file if exists
    instances_to_process = default_instances
    if os.path.exists("all_instances.txt"):
        with open("all_instances.txt", 'r') as f:
            instances_to_process = [line.strip() for line in f if line.strip()]
    
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
        for file in [f'result_{instance_name}_{config_name}.json', f'checkpoint_{instance_name}_{config_name}.json']:
            if os.path.exists(file):
                os.remove(file)
        
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
                        result['Runtime'] = "TIMEOUT"
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

