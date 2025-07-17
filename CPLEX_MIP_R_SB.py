"""
2D Bin Packing Problem - Non-Stacking Version with MIP using CPLEX
With Rotation and Symmetry Breaking constraints
"""

from docplex.mp.model import Model
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
import sys
import signal
import math
import pandas as pd
import timeit
import subprocess
import traceback
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Global variables to track best solution found so far
best_bins = float('inf')
best_assignments = []
best_positions = []
best_rotations = []
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
    
    with open(f'results_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create output folder if it doesn't exist
if not os.path.exists('CPLEX_MIP_R_SB'):
    os.makedirs('CPLEX_MIP_R_SB')

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

def first_fit_upper_bound_with_rotation(rectangles, W, H):
    """First-fit heuristic to get upper bound with rotation consideration"""
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
        
        # Try both orientations (original and rotated)
        orientations = [(w, h), (h, w)] if w != h else [(w, h)]
        
        for ow, oh in orientations:
            if ow > W or oh > H:
                continue
                
            # Try to place in existing bins
            for bin_rects in bins:
                pos = fits(bin_rects, ow, oh, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], ow, oh))
                    placed = True
                    break
            
            if placed:
                break
        
        # If not placed, create a new bin
        if not placed:
            # Try both orientations for new bin
            for ow, oh in orientations:
                if ow <= W and oh <= H:
                    bins.append([(0, 0, ow, oh)])
                    placed = True
                    break
            
            if not placed:
                # Rectangle doesn't fit in any orientation
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
        'Instance': instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }
    
    with open(f'checkpoint_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def display_solution(W, H, rectangles, positions, assignments, rotations, instance_name):
    """Display solution with one subplot per bin, showing rotation status"""
    n_bins = len(set(assignments))
    n_rectangles = len(rectangles)
    
    # Determine layout of subplots
    ncols = min(n_bins, 3)
    nrows = math.ceil(n_bins / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle(f'Solution for {instance_name} - {n_bins} bins (with rotation)', fontsize=16)
    
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
    
    # Use colormap for better visualization
    colors = cm.viridis(np.linspace(0, 1, n_rectangles))
    
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
                
                # Apply rotation if needed
                if rotations[item_idx] > 0.5:
                    width, height = height, width
                
                x, y = positions[item_idx]
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=colors[item_idx],
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Calculate text color based on background brightness
                rgb = mcolors.to_rgb(colors[item_idx])
                brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                text_color = 'white' if brightness < 0.6 else 'black'
                
                # Add item number and rotation status
                rot_info = 'R' if rotations[item_idx] > 0.5 else 'NR'
                ax.text(x + width/2, y + height/2, f'{item_idx + 1}\n{rot_info}', 
                       ha='center', va='center', fontweight='bold', color=text_color)
            
            # Set grid and ticks
            ax.set_xticks(range(0, W+1, max(1, W//10)))
            ax.set_yticks(range(0, H+1, max(1, H//10)))
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'CPLEX_MIP_R_SB/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def solve_bin_packing_with_rotation(W, H, rectangles, lower_bound, upper_bound, time_limit=900):
    """
    Solve 2D Bin Packing using CPLEX MIP with rotation and symmetry breaking
    
    Args:
        W: Width of each bin
        H: Height of each bin
        rectangles: List of (width, height) tuples
        lower_bound: Lower bound on number of bins
        upper_bound: Upper bound on number of bins
        time_limit: Time limit in seconds
    
    Returns:
        Dictionary with solution or None if no solution found
    """
    global best_bins, best_assignments, best_positions, best_rotations
    
    # Create the model
    mdl = Model(name="2D_BinPacking_Rotation_SB")
    
    n = len(rectangles)
    max_bins = min(n, upper_bound)  # No need for more bins than items
    
    print(f"Creating MIP model with {n} items and up to {max_bins} bins...")
    start_model_time = time.time()
    
    # Variables
    # 1. Position variables
    x = {}  # x[i,b] = x-position of item i in bin b
    y = {}  # y[i,b] = y-position of item i in bin b
    for i in range(n):
        for b in range(max_bins):
            x[i,b] = mdl.continuous_var(lb=0, name=f'x_{i}_{b}')
            y[i,b] = mdl.continuous_var(lb=0, name=f'y_{i}_{b}')
    
    # 2. Assignment variables
    z = {}  # z[i,b] = 1 if item i is assigned to bin b
    for i in range(n):
        for b in range(max_bins):
            z[i,b] = mdl.binary_var(name=f'z_{i}_{b}')
    
    # 3. Bin usage variables
    u = {}  # u[b] = 1 if bin b is used
    for b in range(max_bins):
        u[b] = mdl.binary_var(name=f'u_{b}')
    
    # 4. Rotation variables
    rotate = {}  # rotate[i] = 1 if item i is rotated
    for i in range(n):
        rotate[i] = mdl.binary_var(name=f'rotate_{i}')
    
    # 5. Effective width and height after rotation
    w_eff = {}  # Effective width
    h_eff = {}  # Effective height
    for i in range(n):
        w_eff[i] = mdl.continuous_var(name=f'w_eff_{i}')
        h_eff[i] = mdl.continuous_var(name=f'h_eff_{i}')
        
        # Define effective dimensions based on rotation
        mdl.add_constraint(w_eff[i] == rectangles[i][0] * (1 - rotate[i]) + rectangles[i][1] * rotate[i])
        mdl.add_constraint(h_eff[i] == rectangles[i][1] * (1 - rotate[i]) + rectangles[i][0] * rotate[i])
    
    # 6. Auxiliary variables for non-overlapping constraints
    left = {}   # left[i,j,b] = 1 if item i is to the left of item j in bin b
    right = {}  # right[i,j,b] = 1 if item i is to the right of item j in bin b
    below = {}  # below[i,j,b] = 1 if item i is below item j in bin b
    above = {}  # above[i,j,b] = 1 if item i is above item j in bin b
    
    for i in range(n):
        for j in range(i+1, n):
            for b in range(max_bins):
                left[i,j,b] = mdl.binary_var(name=f'left_{i}_{j}_{b}')
                right[i,j,b] = mdl.binary_var(name=f'right_{i}_{j}_{b}')
                below[i,j,b] = mdl.binary_var(name=f'below_{i}_{j}_{b}')
                above[i,j,b] = mdl.binary_var(name=f'above_{i}_{j}_{b}')
    
    # Keep track of constraint count
    constraint_count = 0
    
    # Constraints
    
    # 1. Each item must be placed in exactly one bin
    for i in range(n):
        mdl.add_constraint(mdl.sum(z[i,b] for b in range(max_bins)) == 1)
        constraint_count += 1
    
    # 2. Bin usage constraints
    for b in range(max_bins):
        for i in range(n):
            mdl.add_constraint(z[i,b] <= u[b])
            constraint_count += 1
    
    # 3. Position bounds (considering rotation)
    for i in range(n):
        for b in range(max_bins):
            mdl.add_constraint(x[i,b] + w_eff[i] <= W + (W + H) * (1 - z[i,b]))
            mdl.add_constraint(y[i,b] + h_eff[i] <= H + (W + H) * (1 - z[i,b]))
            constraint_count += 2
    
    # 4. Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        mdl.add_constraint(u[b] <= u[b-1])
        constraint_count += 1
    
    # 5. Symmetry Breaking: Largest rectangle placement
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    if n > 1:
        mdl.add_constraint(z[max_area_idx, 0] == 1)
        constraint_count += 1
        
        # Domain reduction for largest rectangle
        mdl.add_constraint(x[max_area_idx, 0] <= (W - min(rectangles[max_area_idx])) / 2)
        mdl.add_constraint(y[max_area_idx, 0] <= (H - min(rectangles[max_area_idx])) / 2)
        constraint_count += 2
    
    # 6. Non-overlapping constraints with rotation consideration - Fixed version
    for i in range(n):
        for j in range(i+1, n):
            for b in range(max_bins):
                # If both items i and j are in bin b, they must not overlap
                M = W + H  # Big-M value
                
                # At least one separation must be true if both items are in bin b
                mdl.add_constraint(left[i,j,b] + right[i,j,b] + below[i,j,b] + above[i,j,b] >= 
                                  z[i,b] + z[j,b] - 1)
                constraint_count += 1
                
                # Position constraints based on separation choices (with rotation)
                # i to left of j: x[i] + w_eff[i] <= x[j]
                mdl.add_constraint(x[i,b] + w_eff[i] <= x[j,b] + M * (1 - left[i,j,b]))
                
                # i to right of j: x[j] + w_eff[j] <= x[i]  
                mdl.add_constraint(x[j,b] + w_eff[j] <= x[i,b] + M * (1 - right[i,j,b]))
                
                # i below j: y[i] + h_eff[i] <= y[j]
                mdl.add_constraint(y[i,b] + h_eff[i] <= y[j,b] + M * (1 - below[i,j,b]))
                
                # i above j: y[j] + h_eff[j] <= y[i]
                mdl.add_constraint(y[j,b] + h_eff[j] <= y[i,b] + M * (1 - above[i,j,b]))
                constraint_count += 5
    
    # 7. Same-sized rectangles symmetry breaking
    for i in range(n):
        for j in range(i+1, n):
            if rectangles[i][0] == rectangles[j][0] and rectangles[i][1] == rectangles[j][1]:
                # For identical rectangles, apply ordering
                for b in range(max_bins):
                    for b2 in range(b):
                        mdl.add_constraint(z[i,b] + z[j,b2] <= 1)
                        constraint_count += 1
                
                # Lexicographic ordering in same bin
                for b in range(max_bins):
                    mdl.add_constraint(left[i,j,b] + below[i,j,b] >= z[i,b] + z[j,b] - 1)
                    constraint_count += 1
    
    # 8. One pair constraint - Fixed version
    if n >= 2:
        # Fix relative position of first two rectangles
        for b in range(max_bins):
            # Ensure rectangle 0 is to the left of or below rectangle 1
            mdl.add_constraint(right[0,1,b] == 0)  # Rectangle 0 cannot be to the right of rectangle 1
            mdl.add_constraint(above[0,1,b] == 0)  # Rectangle 0 cannot be above rectangle 1
            constraint_count += 2
    
    # Set objective: minimize number of bins used
    mdl.minimize(mdl.sum(u[b] for b in range(max_bins)))
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Set time limit
    mdl.set_time_limit(time_limit)
    
    # Solve
    print("Solving model...")
    solve_start = time.time()
    solution = mdl.solve(log_output=True)
    solve_time = time.time() - solve_start
    
    print(f"Solver finished in {solve_time:.2f}s with status: {mdl.solve_details.status}")
    
    if solution:
        # Count bins actually used
        bins_used = sum(1 for b in range(max_bins) if solution.get_value(u[b]) > 0.5)
        
        # Update best solution
        if bins_used < best_bins:
            best_bins = bins_used
            
            # Extract item assignments, positions, and rotations
            assignments = [-1] * n
            positions = [(0, 0)] * n
            rotations = [0] * n
            
            for i in range(n):
                rotations[i] = solution.get_value(rotate[i])
                for b in range(max_bins):
                    if solution.get_value(z[i,b]) > 0.5:
                        assignments[i] = b
                        positions[i] = (solution.get_value(x[i,b]), solution.get_value(y[i,b]))
                        break
            
            best_assignments = assignments.copy()
            best_positions = positions.copy()
            best_rotations = rotations.copy()
            
            # Save checkpoint with solution
            save_checkpoint(instance_id, best_bins)
        
        result = {
            'status': 'OPTIMAL' if 'optimal' in mdl.solve_details.status else 'FEASIBLE',
            'n_bins': bins_used,
            'assignments': assignments,
            'positions': positions,
            'rotations': rotations,
            'solve_time': solve_time,
            'best_bound': mdl.solve_details.best_bound,
            'gap': mdl.solve_details.mip_relative_gap * 100 if mdl.solve_details.mip_relative_gap is not None else None
        }
        
        return result
    else:
        print("No solution found")
        return None
start = timeit.default_timer()

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create output folder if it doesn't exist
        if not os.path.exists('CPLEX_MIP_R_SB'):
            os.makedirs('CPLEX_MIP_R_SB')
        
        # Read existing Excel file to check completed instances
        excel_file = 'CPLEX_MIP_R_SB.xlsx'
        if os.path.exists(excel_file):
            try:
                existing_df = pd.read_excel(excel_file)
                completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
            except:
                existing_df = pd.DataFrame()
                completed_instances = [
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
        else:
            existing_df = pd.DataFrame()
            completed_instances = []
        # Set timeout
        TIMEOUT = 900  # 30 minutes
        
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
            for temp_file in [f'results_{instance_id}.json', f'checkpoint_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -t {TIMEOUT} python3 CPLEX_MIP_R_SB.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_{instance_id}.json'):
                    with open(f'results_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_{instance_id}.json'):
                    with open(f'checkpoint_{instance_id}.json', 'r') as f:
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
                        except:
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
            for temp_file in [f'results_{instance_id}.json', f'checkpoint_{instance_id}.json']:
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
            best_rotations = []
            
            # Read input
            input_data = read_file_instance(instance_name)
            n_items = int(input_data[0])
            bin_size = input_data[1].split()
            W = int(bin_size[0])
            H = int(bin_size[1])
            rectangles = [[int(val) for val in line.split()] for line in input_data[2:2 + n_items]]
            
            # Calculate bounds
            lower_bound = calculate_lower_bound(rectangles, W, H)
            upper_bound = min(n_items, first_fit_upper_bound_with_rotation(rectangles, W, H))
            
            print(f"Solving 2D Bin Packing with CPLEX MIP (with rotation and symmetry breaking) for instance {instance_name}")
            print(f"Bin size: {W}x{H}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with MIP
            solution = solve_bin_packing_with_rotation(W, H, rectangles, lower_bound, upper_bound, time_limit=900)
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            # Process result
            if solution:
                n_bins = solution['n_bins']
                status = 'OPTIMAL' if solution['status'] == 'OPTIMAL' else 'FEASIBLE'
                
                # Display solution
                display_solution(W, H, rectangles, solution['positions'], solution['assignments'], 
                               solution['rotations'], instance_name)
                
                print(f"Solution found: {n_bins} bins, Status: {status}")
                if solution['gap'] is not None:
                    print(f"Optimality gap: {solution['gap']:.2f}%")
            else:
                n_bins = best_bins if best_bins != float('inf') else upper_bound
                status = 'ERROR'
                print(f"No solution found. Using best bound: {n_bins}")
            
            # Create result
            result = {
                'Instance': instance_name,
                'Runtime': runtime,
                'N_Bins': n_bins,
                'Status': status,
            }
            
            # Save to Excel
            excel_file = 'CPLEX_MIP_R_SB.xlsx'
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
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save JSON result for controller
            with open(f'results_{instance_id}.json', 'w') as f:
                json.dump(result, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {n_bins}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()
            
            # Create error result
            result = {
                'Instance': instance_name,
                'Runtime': timeit.default_timer() - start,
                'N_Bins': best_bins if best_bins != float('inf') else upper_bound,
                'Status': 'ERROR',
            }
            
            # Save error result to Excel
            excel_file = 'CPLEX_MIP_R_SB.xlsx'
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
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            with open(f'results_{instance_id}.json', 'w') as f:
                json.dump(result, f)