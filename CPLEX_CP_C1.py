"""
2D Bin Packing Problem - Non-Stacking Version with CP using CPLEX
Symmetry Breaking: C1 (Bins are used in order)
"""

from docplex.cp.model import CpoModel
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import timeit
import sys
import signal
import os
import json
import subprocess
import traceback

# Global variables to track best solution found so far
best_bins = float('inf')
best_assignments = []
best_positions = []
upper_bound = 0

# Signal handler for graceful interruption
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    current_bins = best_bins if best_bins != float('inf') else upper_bound
    print(f"Best bins found before interrupt: {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': current_bins,
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_CPLEX_CP_C1_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create output folder if it doesn't exist
if not os.path.exists('CPLEX_CP_C1'):
    os.makedirs('CPLEX_CP_C1')

def read_file_instance(instance_name):
    """Read instance file based on instance name"""
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

instances = [
    "",
    # BENG instances (10 instances)
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
    "CL_10_100_6", "CL_10_100_7", "CL_10_100_8", "CL_10_100_9", "CL_10_100_10"
]

def first_fit_upper_bound(rectangles, W, H):
    """First-fit heuristic to get upper bound"""
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
        w, h = rect[0], rect[1]
        placed = False
        
        # Try to place in existing bins
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break
        
        # If not placed, create a new bin
        if not placed:
            if w <= W and h <= H:
                bins.append([(0, 0, w, h)])
            else:
                # Rectangle doesn't fit in any bin
                return float('inf')
    
    return len(bins)

def calculate_lower_bound(rectangles, W, H):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = W * H
    return math.ceil(total_area / bin_area)

def save_checkpoint(instance_id, bins, status="IN_PROGRESS"):
    """Save checkpoint for current progress"""
    checkpoint = {
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }
    
    with open(f'checkpoint_CPLEX_CP_C1_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def solve_bin_packing(W, H, rectangles, time_limit=900):
    """
    Solve 2D Bin Packing using CPLEX CP with C1 symmetry breaking and objective minimization
    
    Args:
        W: Width of each bin
        H: Height of each bin
        rectangles: List of (width, height) tuples
        time_limit: Time limit in seconds
        
    Returns:
        Dictionary with solution or None if no solution found
    """
    global best_bins, best_assignments, best_positions, upper_bound
    
    # Create the CP model
    model = CpoModel(name="2D_BinPacking_C1")
    
    n = len(rectangles)
    
    # Calculate upper bound
    ub = min(n, first_fit_upper_bound(rectangles, W, H))
    upper_bound = ub
    max_bins = ub  # Use upper bound as maximum number of bins
    
    print(f"Creating CPLEX CP model with {n} items and up to {max_bins} bins...")
    start_model_time = time.time()
    
    # Variables: bin assignment and coordinates for each rectangle
    bin_assignment = [model.integer_var(0, max_bins-1, f'bin_{i}') for i in range(n)]
    x = {}
    y = {}
    
    for i in range(n):
        for b in range(max_bins):
            x[i, b] = model.integer_var(0, W - rectangles[i][0], f'x_{i}_{b}')
            y[i, b] = model.integer_var(0, H - rectangles[i][1], f'y_{i}_{b}')
    
    # Bin usage variables
    bin_used = [model.binary_var(f'used_{b}') for b in range(max_bins)]
    
    # Link bin usage with assignments
    for b in range(max_bins):
        model.add(model.if_then(
            model.logical_or([bin_assignment[i] == b for i in range(n)]),
            bin_used[b] == 1
        ))
    
    # C1 Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        model.add(bin_used[b-1] >= bin_used[b])
    
    # Find largest rectangle for additional C1 symmetry breaking
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    # C1 Additional Symmetry Breaking: Place largest rectangle in first bin
    if n > 1:
        model.add(bin_assignment[max_area_idx] == 0)
        
        # Position the largest rectangle in the bottom-left quadrant
        model.add(x[max_area_idx, 0] <= (W - rectangles[max_area_idx][0]) // 2)
        model.add(y[max_area_idx, 0] <= (H - rectangles[max_area_idx][1]) // 2)
    
    # Non-overlapping constraints for rectangles in the same bin
    for i in range(n):
        for j in range(i+1, n):
            for b in range(max_bins):
                # If both rectangles are in bin b, they must not overlap
                model.add(model.logical_or([
                    bin_assignment[i] != b,
                    bin_assignment[j] != b,
                    x[i, b] + rectangles[i][0] <= x[j, b],  # i is to the left of j
                    x[j, b] + rectangles[j][0] <= x[i, b],  # j is to the left of i
                    y[i, b] + rectangles[i][1] <= y[j, b],  # i is below j
                    y[j, b] + rectangles[j][1] <= y[i, b]   # j is below i
                ]))
                
                # C1 Symmetry Breaking - Large rectangles
                if rectangles[i][0] + rectangles[j][0] > W:
                    # If two rectangles can't fit side by side, disable horizontal placements
                    model.add(model.logical_or([
                        bin_assignment[i] != b,
                        bin_assignment[j] != b,
                        y[i, b] + rectangles[i][1] <= y[j, b],  # i is below j
                        y[j, b] + rectangles[j][1] <= y[i, b]   # j is below i
                    ]))
                
                if rectangles[i][1] + rectangles[j][1] > H:
                    # If two rectangles can't fit vertically, disable vertical placements
                    model.add(model.logical_or([
                        bin_assignment[i] != b,
                        bin_assignment[j] != b,
                        x[i, b] + rectangles[i][0] <= x[j, b],  # i is to the left of j
                        x[j, b] + rectangles[j][0] <= x[i, b]   # j is to the left of i
                    ]))
    
    # C1 Symmetry Breaking - For identical rectangles, enforce ordering
    for i in range(n):
        for j in range(i+1, n):
            if rectangles[i][0] == rectangles[j][0] and rectangles[i][1] == rectangles[j][1]:
                # If rectangles are identical, enforce bin ordering
                model.add(bin_assignment[i] <= bin_assignment[j])
                
                # If in same bin, enforce positional ordering (lexicographic)
                for b in range(max_bins):
                    model.add(model.logical_or([
                        bin_assignment[i] != b,
                        bin_assignment[j] != b,
                        x[i, b] < x[j, b],  # i is strictly to the left of j
                        model.logical_and([x[i, b] == x[j, b], y[i, b] <= y[j, b]])  # or same x but i is at or below j
                    ]))
    
    # Set objective: minimize the number of bins used
    model.add(model.minimize(model.sum(bin_used)))
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Solve with time limit
    print("Solving with CPLEX CP...")
    solve_start = time.time()
    
    try:
        solution = model.solve(TimeLimit=time_limit, LogVerbosity='Quiet')
        solve_time = time.time() - solve_start
        print(f"Solver finished in {solve_time:.2f}s")
        
    except Exception as e:
        solve_time = time.time() - solve_start
        print(f"Solver error: {str(e)}")
        print(f"Solver interrupted after {solve_time:.2f}s")
        
        # Return current best solution on error
        return {
            'status': 'ERROR',
            'n_bins': best_bins if best_bins != float('inf') else upper_bound,
            'assignments': best_assignments,
            'positions': best_positions,
            'solve_time': solve_time,
            'objective_value': best_bins if best_bins != float('inf') else upper_bound
        }
    
    if solution and solution.is_solution():
        # Extract solution
        used_bins = sum(1 for b in range(max_bins) if solution.get_value(bin_used[b]) > 0.5)
        assignments = [solution.get_value(bin_assignment[i]) for i in range(n)]
        
        positions = []
        for i in range(n):
            b = assignments[i]
            positions.append((solution.get_value(x[i, b]), solution.get_value(y[i, b])))
        
        # Update global best solution
        best_bins = used_bins
        best_assignments = assignments.copy()
        best_positions = positions.copy()
        
        # Save final checkpoint
        save_checkpoint(instance_id, best_bins)
        
        print(f"Solution found: {used_bins} bins")
        
        return {
            'status': 'OPTIMAL' if solution.get_solve_status() == 'Optimal' else 'FEASIBLE',
            'n_bins': used_bins,
            'assignments': assignments,
            'positions': positions,
            'solve_time': solution.get_solve_time(),
            'objective_value': used_bins
        }
    else:
        print("No solution found")
        return {
            'status': 'NO_SOLUTION',
            'n_bins': best_bins if best_bins != float('inf') else upper_bound,
            'assignments': best_assignments,
            'positions': best_positions,
            'solve_time': solve_time,
            'objective_value': None
        }

def display_solution(W, H, rectangles, positions, assignments, instance_name):
    """Display solution with one subplot per bin"""
    n_bins = len(set(assignments))
    n_rectangles = len(rectangles)
    
    # Determine layout of subplots
    ncols = min(n_bins, 3)
    nrows = math.ceil(n_bins / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle(f'Solution for {instance_name} - {n_bins} bins', fontsize=16)
    
    # Handle different subplot configurations
    if n_bins == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Create bins structure
    bins = [[] for _ in range(n_bins)]
    for i in range(n_rectangles):
        bins[assignments[i]].append(i)
    
    # Draw rectangles for each bin
    for bin_idx, items in enumerate(bins):
        if bin_idx < len(axes):
            ax = axes[bin_idx]
            ax.set_title(f'Bin {bin_idx + 1}')
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect('equal')
            
            # Draw each rectangle in this bin
            for item_idx in items:
                width, height = rectangles[item_idx]
                x, y = positions[item_idx]
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=plt.cm.Set3(item_idx % 12),
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Add item number
                ax.text(x + width/2, y + height/2, str(item_idx + 1), 
                       ha='center', va='center', fontweight='bold')
            
            # Set grid and ticks
            ax.set_xticks(range(0, W+1, max(1, W//10)))
            ax.set_yticks(range(0, H+1, max(1, H//10)))
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'CPLEX_CP_C1/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create output folder if it doesn't exist
        if not os.path.exists('CPLEX_CP_C1'):
            os.makedirs('CPLEX_CP_C1')
        
        # Read existing Excel file to check completed instances
        excel_file = 'CPLEX_CP_C1.xlsx'
        if os.path.exists(excel_file):
            try:
                existing_df = pd.read_excel(excel_file)
                completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
            except Exception as e:
                print(f"Error reading Excel file: {e}. Starting with empty DataFrame.")
                existing_df = pd.DataFrame()
                completed_instances = []
        else:
            existing_df = pd.DataFrame()
            completed_instances = []
        
        # Set timeout
        TIMEOUT = 900  
        
        # Start from instance 1 (skip index 0 which is empty)
        for instance_id in range(1, len(instances)):
            instance_name = instances[instance_id]
            
            # Skip if already completed
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
                
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_CPLEX_CP_C1_{instance_id}.json', f'checkpoint_CPLEX_CP_C1_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 CPLEX_CP_C1.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_CPLEX_CP_C1_{instance_id}.json'):
                    with open(f'results_CPLEX_CP_C1_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_CPLEX_CP_C1_{instance_id}.json'):
                    with open(f'checkpoint_CPLEX_CP_C1_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                    print(f"Instance {instance_name} timed out. Using checkpoint data.")
                
                # Process results
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Bins used: {result['N_Bins']}, Runtime: {result['Runtime']}")
                    
                    # Update Excel
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                            
                            if instance_exists:
                                instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                                for key, value in result.items():
                                    existing_df.at[instance_idx, key] = value
                            else:
                                result_df = pd.DataFrame([result])
                                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        except Exception as e:
                            print(f"Error reading Excel file: {e}. Creating new DataFrame.")
                            existing_df = pd.DataFrame([result])
                    else:
                        existing_df = pd.DataFrame([result])
                    
                    existing_df.to_excel(excel_file, index=False)
                    print(f"Results saved to {excel_file}")
                else:
                    print(f"No results found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up temp files
            for temp_file in [f'results_CPLEX_CP_C1_{instance_id}.json', f'checkpoint_CPLEX_CP_C1_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        instance_name = instances[instance_id]
        
        start = timeit.default_timer()
        
        try:
            print(f"\nProcessing instance {instance_name}")
            
            # Reset global variables
            best_bins = float('inf')
            best_assignments = []
            best_positions = []
            
            # Read input
            input_data = read_file_instance(instance_name)
            n_items = int(input_data[0])
            bin_size = input_data[1].split()
            W = int(bin_size[0])
            H = int(bin_size[1])
            rectangles = [[int(val) for val in line.split()] for line in input_data[2:2 + n_items]]
            
            # Calculate bounds
            lower_bound = calculate_lower_bound(rectangles, W, H)
            upper_bound = min(n_items, first_fit_upper_bound(rectangles, W, H))
            
            print(f"Solving 2D Bin Packing with CPLEX CP and C1 symmetry breaking for instance {instance_name}")
            print(f"Bin size: {W}x{H}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with CP
            try:
                result = solve_bin_packing(W, H, rectangles, time_limit=900)
            except Exception as e:
                print(f"Error in instance {instance_name}: {str(e)}")
                # Create error result
                result = {
                    'status': 'ERROR',
                    'n_bins': upper_bound,
                    'assignments': [],
                    'positions': [],
                    'solve_time': 0,
                    'objective_value': upper_bound
                }
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            # Process result
            if result and len(result['positions']) > 0:
                # Display solution
                display_solution(W, H, rectangles, result['positions'], result['assignments'], instance_name)
                
                print(f"Solution found: {result['n_bins']} bins")
                status = result['status']
                
                if result['objective_value'] is not None:
                    print(f"Objective value: {result['objective_value']}")
            else:
                print("No feasible solution found.")
                status = 'ERROR'
            
            # Create result
            result_data = {
                'Instance': instance_name,
                'Runtime': runtime,
                'N_Bins': result['n_bins'] if result else (best_bins if best_bins != float('inf') else upper_bound),
                'Status': status if result else 'ERROR'
            }
            
            # Save to Excel
            excel_file = 'CPLEX_CP_C1.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result_data.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result_data])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except Exception as e:
                    print(f"Error reading Excel file: {e}. Creating new DataFrame.")
                    existing_df = pd.DataFrame([result_data])
            else:
                existing_df = pd.DataFrame([result_data])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save JSON result for controller
            with open(f'results_CPLEX_CP_C1_{instance_id}.json', 'w') as f:
                json.dump(result_data, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {result_data['N_Bins']}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()
            
            # Create error result
            result_data = {
                'Instance': instance_name,
                'Runtime': timeit.default_timer() - start,
                'N_Bins': best_bins if best_bins != float('inf') else upper_bound,
                'Status': 'ERROR'
            }
            
            # Save error result to Excel
            excel_file = 'CPLEX_CP_C1.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result_data.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result_data])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except Exception as e:
                    print(f"Error reading Excel file: {e}. Creating new DataFrame.")
                    existing_df = pd.DataFrame([result_data])
            else:
                existing_df = pd.DataFrame([result_data])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            with open(f'results_CPLEX_CP_C1_{instance_id}.json', 'w') as f:
                json.dump(result_data, f)