import math
import os
import signal
import subprocess
import logging

import pandas as pd
from pysat.formula import CNF
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt

from pysat.solvers import Glucose3
import datetime
import pandas as pd
import os
import sys
import time
from datetime import datetime
import json
import timeit
import numpy as np


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
config_name = "BPP_S_R"
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
        'Optimal_Bins': current_bins,
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
def read_file_instance(instance_name):
    """Read instance file similar to SPP format"""
    # Try different input folders similar to SPP
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
        'Optimal_Bins': bins if bins != float('inf') else upper_bound,
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
    """Write results to Excel file - overwrite if instance exists"""
    try:
        # Try to load existing Excel file
        if os.path.exists(excel_file):
            df_existing = pd.read_excel(excel_file, engine='openpyxl')
            
            # Check if instance already exists
            if 'Instance' in df_existing.columns:
                instance_name = results['Instance']
                
                # Find if instance exists and update it
                if instance_name in df_existing['Instance'].values:
                    # Update existing row
                    mask = df_existing['Instance'] == instance_name
                    for key, value in results.items():
                        if key in df_existing.columns:
                            df_existing.loc[mask, key] = value
                        else:
                            df_existing[key] = df_existing.get(key, '')
                            df_existing.loc[mask, key] = value
                    df_combined = df_existing
                    logger.info(f"Updated existing instance {instance_name} in Excel")
                else:
                    # Add new row
                    df_new = pd.DataFrame([results])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    logger.info(f"Added new instance {instance_name} to Excel")
            else:
                # No Instance column, just append
                df_new = pd.DataFrame([results])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # Create new file
            df_combined = pd.DataFrame([results])
            logger.info(f"Created new Excel file with instance {results['Instance']}")
        
        # Save to Excel
        df_combined.to_excel(excel_file, index=False, engine='openpyxl')
        logger.info(f"Results written to Excel: {excel_file}")
        
    except Exception as e:
        logger.error(f"Error writing to Excel: {e}")


def run_with_runlim(command, timeout, memory_limit):
    """Run command with runlim for proper resource management"""
    runlim_cmd = f"./runlim -r {timeout} {command}"
    logger.info(f"Running with runlim: {runlim_cmd}")
    
    try:
        result = subprocess.run(runlim_cmd.split(), 
                              capture_output=True, 
                              text=True, 
                              timeout=timeout + 60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Timeout"  # 124 is timeout exit code

def calculate_lower_bound(rectangles, W, H):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = W * H
    area_lower_bound = math.ceil(total_area / bin_area)
    
    # Check for items that are too large
    for w, h in rectangles:
        if w > W or h > H:
            return float('inf')  # Infeasible
    
    return max(1, area_lower_bound)

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

def BPP_Stacking(n_bins, W, H):
    # Define the variables
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot
    global upper_bound, is_timeout
    height = n_bins * H
    width = W
    cnf = CNF()
    variables = {}
    id_variables = 1
    start = time.time()
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = id_variables  # lri,rj
                id_variables += 1
                variables[f"ud{i + 1},{j + 1}"] = id_variables  # uri,rj
                id_variables += 1
        for e in range(width):  # e+1 because we need to check px[i,e] and px[i,e+1]
            variables[f"px{i + 1},{e}"] = id_variables  # pxi,e
            id_variables += 1
        for f in range(height):  # f+1 because we need to check py[i,f] and py[i,f+1]
            variables[f"py{i + 1},{f}"] = id_variables  # pyi,f
            id_variables += 1

    # Rotated variables
    for i in range(len(rectangles)):
        variables[f"r{i + 1}"] = id_variables
        id_variables += 1
    
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
                

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            non_overlapping(False, i, j, True, True, True, True)
            non_overlapping(True, i, j, True, True, True, True)

    # # Domain encoding to ensure every rectangle stays inside strip's boundary
    for i in range(len(rectangles)):
            if rectangles[i][0] > width: #if rectangle[i]'s width larger than strip's width, it has to be rotated
                cnf.append([variables[f"r{i + 1}"]])
            else:
                pos_x = width - rectangles[i][0]
                if pos_x >= 0:
                    cnf.append([variables[f"r{i + 1}"],
                                        variables[f"px{i + 1},{pos_x}"]])
       
            if rectangles[i][1] > height:
                cnf.append([variables[f"r{i + 1}"]])
            else:
                pos_y = height - rectangles[i][1]
                if pos_y >= 0:
                    cnf.append([variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{pos_y}"]])

            # Rotated
            if rectangles[i][1] > width:
                cnf.append([-variables[f"r{i + 1}"]])
            else:
                pos_x = width - rectangles[i][1]
                if pos_x >= 0:
                    cnf.append([-variables[f"r{i + 1}"],
                                        variables[f"px{i + 1},{pos_x}"]])
            if rectangles[i][0] > height:
                cnf.append([-variables[f"r{i + 1}"]])
            else:
                pos_y = height - rectangles[i][0]
                if pos_y >= 0:
                    cnf.append([-variables[f"r{i + 1}"],
                                    variables[f"py{i + 1},{pos_y}"]])
    
    for k in range(1, n_bins):
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
                h = rectangles[i][0]
                cnf.append([-variables[f"r{i + 1}"], variables[f"py{i + 1},{k * H - h}"],
                            -variables[f"py{i + 1},{k * H - 1}"]])
                cnf.append([-variables[f"r{i + 1}"], -variables[f"py{i + 1},{k * H - h}"],
                            variables[f"py{i + 1},{k * H - 1}"]])
    stop = time.time()
    print("Encoding Time:", format(stop-start, ".6f"))  
    variables_length = len(variables)
    clauses_length = len(cnf.clauses)
    save_checkpoint(instance_name, variables_length, clauses_length, best_bins, "IN_PROGRESS")      
    with Glucose3(use_timer=True, bootstrap_with=cnf) as solver: 

        # Solve with timeout check
        try:
            if solver.solve_limited(expect_interrupt=True):
                # Rest of your solver success code...
                pos = [[0 for i in range(2)] for j in range(len(rectangles))]
                rotation = []
                model = solver.get_model()
                solver_time = format(time.time() - start, ".3f")
                print("Solver time:", solver_time)
                print("SAT")
                if n_bins < best_bins:
                    best_bins = n_bins
                    print(f"New best bins found: {best_bins}")
                    save_checkpoint(instance_name, variables_length, clauses_length, best_bins, "SOLUTION_FOUND")
                result = {}
                for var in model:
                    if var > 0:
                        result[list(variables.keys())[list(variables.values()).index(var)]] = True
                    else:
                        result[list(variables.keys())[list(variables.values()).index(-var)]] = False

                for i in range(len(rectangles)):
                    rotation.append(result[f"r{i + 1}"])
                    for e in range(width - 1):
                        if result[f"px{i + 1},{e}"] == False and result[f"px{i + 1},{e + 1}"] == True:
                            pos[i][0] = e + 1
                        if e == 0 and result[f"px{i + 1},{e}"] == True:
                            pos[i][0] = 0
                    for f in range(height - 1):
                        if result[f"py{i + 1},{f}"] == False and result[f"py{i + 1},{f + 1}"] == True:
                            pos[i][1] = f + 1
                        if f == 0 and result[f"py{i + 1},{f}"] == True:
                            pos[i][1] = 0  
                return ["sat", pos, rotation]
            else:
                print("UNSAT")
                return ["unsat", [], []]
        except:
            raise
            
def BPP_linear_search(lower, upper):
    global optimal_bins, optimal_pos, optimal_rot
    for n in range(lower, upper + 1):
        print(f"Trying with {n} bins")
        result = BPP_Stacking(n, W, H)
        if result[0] == "sat":
                optimal_bins = n
                print(f"New best bins found: {best_bins}")
                optimal_pos = result[1]
                optimal_rot = result[2]
                print(f"Optimal positions before: {optimal_pos}")
                for i in range(len(optimal_pos)):
                    optimal_pos[i].append(optimal_pos[i][1] // H) # append the bin number
                    optimal_pos[i][1] = optimal_pos[i][1] % H
                return
        else:
            print(f"Found solution with {n} bins, but not better than current best {best_bins}")
   

# Modify the BPP_binary_search function to check timeout
def BPP_binary_search(lower, upper):
    global bins_used
    if (lower <= upper):
        mid = (lower + upper) // 2
        print(f"Trying with {mid} bins")
        result = BPP_Stacking(mid, W, H) 
        # Rest of the function stays the same
        if result[0] == "unsat":
            return BPP_binary_search(mid + 1, upper)
        else:
            print(f"Optimal solution found with {mid} bins")
            global optimal_bins, optimal_pos, optimal_rot
            optimal_bins = mid
            optimal_pos = result[1]
            print(f"Optimal positions before: {optimal_pos}")
            optimal_rot = result[2]
            bins_used = [[i for i in range(n_items) if optimal_pos[i][1] // H == j] for j in range(mid)]
            for i in range(len(optimal_pos)):
                # optimal_pos[i].append(optimal_pos[i][1] // H) # append the bin number
                optimal_pos[i][1] = optimal_pos[i][1] % H
            return BPP_binary_search(lower, mid - 1)
    else:
        return -1
    

def display_solution(bins, rectangles, positions, instance_name, W=None, H=None):
    """Display solution with bins visualization"""
    if W is None or H is None:
        # Estimate bin dimensions if not provided
        W = max(w for w, h in rectangles) if rectangles else 100
        H = max(h for w, h in rectangles) if rectangles else 100
    
    for bin_idx, bin_items in enumerate(bins):
        if not bin_items:
            continue
            
        fig, ax = plt.subplots()
        ax = plt.gca()
        plt.title(f"Bin {bin_idx + 1} - {instance_name}")

        for item_idx in bin_items:
            rect = plt.Rectangle(positions[item_idx], *rectangles[item_idx], 
                               edgecolor="#333", alpha=0.7,
                               facecolor=plt.cm.Set3(item_idx % 12))
            ax.add_patch(rect)
            # Add item number
            ax.text(positions[item_idx][0] + rectangles[item_idx][0]/2, 
                   positions[item_idx][1] + rectangles[item_idx][1]/2, 
                   str(item_idx+1), ha='center', va='center', fontsize=8)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xlabel('width')
        ax.set_ylabel('height')
        # Save the plot to BPP folder
        plt.savefig(f'{config_name}/{instance_name}_bin_{bin_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()

def display_solution_each_bin(W, H, rectangles, positions, rotations, bins_used, instance_name):
    """Display all bins in one window with subplots"""
    # # Use the new colormap API for compatibility
    # cmap = matplotlib.colormaps.get_cmap('tab20')
    # colors = [cmap(i % cmap.N) for i in range(len(rectangles))]
    for bin_idx, bin_items in enumerate(bins_used):
        if not bin_items:
            continue
            
        fig, ax = plt.subplots()
        ax = plt.gca()
        plt.title(f"Bin {bin_idx + 1} - {instance_name}")
    n_bins = len(bins_used)
    if n_bins == 0:
        logger.warning("No bins used in solution")
        return
        
    ncols = min(n_bins, 4)
    nrows = (n_bins + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    fig.suptitle(f"BPP_S_R - {instance_name} - {n_bins} bins", fontsize=16, fontweight='bold')
    
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
        ax.set_title(f'Bin {bin_idx + 1}', fontsize=10)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_xlabel('Width', fontsize=8)
        ax.set_ylabel('Height', fontsize=8)
        
        # Draw rectangles for items in this bin
        for item_idx in items_in_bin:
            w, h = rectangles[item_idx]
            if rotations[item_idx]:
                w, h = h, w
            x0, y0 = positions[item_idx]
            
            rect_patch = plt.Rectangle((x0, y0), w, h, 
                                     edgecolor='black', 
                                     facecolor="lightblue", 
                                     alpha=0.7)
            ax.add_patch(rect_patch)
            
            # Add item number and rotation info in the center
            cx, cy = x0 + w/2, y0 + h/2
            
            
            rot_info = 'R' if rotations[item_idx] else ""
            ax.text(cx, cy, f'{item_idx + 1}\n{rot_info}', 
                   ha='center', va='center', 
                    fontweight='bold', fontsize=8)
        
        # Set grid and ticks with small font
        ax.set_xticks(range(0, W+1, max(1, W//10)))
        ax.set_yticks(range(0, H+1, max(1, H//10)))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save plot instead of showing for better compatibility
    try:
        plt.savefig(f'{config_name}/{instance_name}_.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Solution visualization saved to {config_name}/{instance_name}_all_bins.png")
    except Exception as e:
        logger.error(f"Could not save plot: {e}")
        plt.close()


def solve_single_instance(instance_name, timeout=900):
    """Solve a single BPP instance"""
    global start, variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot
    global n_items, W, H, rectangles, upper_bound
    logger.info(f"Solving instance: {instance_name}")
    start = timeit.default_timer()
    
    try:
        # Read input
        lines = read_file_instance(instance_name)
        n_items = int(lines[0])
        W, H = map(int, lines[1].split())
        rectangles = []
        
        for i in range(2, 2 + n_items):
            w, h = map(int, lines[i].split())
            rectangles.append([w, h])
        
        logger.info(f"Items: {n_items}, Bin size: {W}×{H}")
        print(f"Number of items: {n_items}, Width: {W}, Height: {H}")
        print(f"Rectangles: {rectangles}")
        
        # Reset global variables
        optimal_bins = 0
        best_bins = 0
        optimal_pos = []
        optimal_rot = []
        
        # Check for checkpoint
        checkpoint = load_checkpoint(instance_name)
        # Calculate bounds
        lower_bound = calculate_lower_bound(rectangles, W, H)
        print(f"Lower bound: {lower_bound}")
        upper_bound = first_fit_upper_bound(rectangles, W, H)
        print(f"Upper bound (First Fit): {upper_bound}")
        best_bins = upper_bound
        
        # Solve
        BPP_binary_search(lower_bound, upper_bound)
        
        runtime = timeit.default_timer() - start
        
        if optimal_bins > 0 and optimal_pos and optimal_rot:
            status = "COMPLETE"
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
        result = {
            'Instance': instance_name,
            'Variables': variables_length,
            'Clauses': clauses_length,
            'Runtime': runtime,
            'Optimal_Bins': n_bins,
            'Status': status,
        }
        
        # Save JSON result for controller
        json_file = f'result_{instance_name}_{config_name}.json'
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {json_file}")
        
        # Save to Excel
        excel_file = f'{config_name}.xlsx'
        write_excel_results(result, excel_file)

        for file in [f'result_{instance_name}_{config_name}.json', f'checkpoint_{instance_name}_{config_name}.json']:
            if os.path.exists(file):
                os.remove(file)
        # Display solution if found
        display_solution_each_bin(W, H, rectangles, optimal_pos, optimal_rot, bins_used, instance_name)
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving {instance_name}: {e}")
        runtime = timeit.default_timer() - start
        
        # Save error result
        current_bins = best_bins if best_bins != float('inf') else upper_bound
        result = {
            'Instance': instance_name,
            'Variables': variables_length,
            'Clauses': clauses_length,
            'Runtime': runtime,
            'Optimal_Bins': current_bins,
            'Status': 'ERROR'
        }
        
        # Save error result
        with open(f'result_{instance_name}_{config_name}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        excel_file = f'{config_name}.xlsx'
        write_excel_results(result, excel_file)
        for file in [f'result_{instance_name}_{config_name}.json', f'checkpoint_{instance_name}_{config_name}.json']:
            if os.path.exists(file):
                os.remove(file)
        return None

def controller_mode():
    """Controller mode - batch process multiple instances"""
    logger.info("Starting BPP controller mode")
    
    # Default instances to process
    default_instances = [
        "BENG01", "BENG02", "BENG03", "BENG04", "BENG05",
        "BENG06", "BENG07", "BENG08", "BENG09", "BENG10"
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
    TIMEOUT = 900  # 15 minutes timeout

    for instance_name in instances_to_process:
        # Skip if already completed
        if instance_name in completed_instances:
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
                command = f"python3 {script_path} {instance_name}"
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
                    if 'Optimal_Bins' in result:
                        logger.info(f"Optimal_Bins: {result['Optimal_Bins']}, Runtime: {result['Runtime']}")
                    
                    # Write to Excel if it's a timeout result
                    if result['Status'] == 'TIMEOUT':
                        write_excel_results(result, excel_file)
                        
            else:
                # Run directly without runlim
                result = solve_single_instance(instance_name, timeout=TIMEOUT)
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
        solve_single_instance(instance_name, timeout=900)
    else:
        print(f"Usage:")
        print(f"  Controller mode: python {sys.argv[0]}")
        print(f"  Single instance: python {sys.argv[0]} <instance_name>")
        print(f"Examples:")
        print(f"  python {sys.argv[0]} BENG01")
        print(f"  python {sys.argv[0]} CL_1_20_1")
        sys.exit(1)