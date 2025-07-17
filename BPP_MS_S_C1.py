import math
import os
import sys
import signal
import json
import subprocess
import time
import tempfile
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import traceback

# Global variables to track best solution found so far
best_num_bins = float('inf')
best_solution = None
variables_length = 0
clauses_length = 0
upper_bound = 0

# Signal handler for graceful interruption
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
    print(f"Best number of bins found before interrupt: {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],
        'Variables': variables_length,
        'Clauses': clauses_length,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': current_bins,
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create BPP_MS_S_C1 folder if it doesn't exist
if not os.path.exists('BPP_MS_S_C1'):
    os.makedirs('BPP_MS_S_C1')

def display_solution(bin_width, bin_height, rectangles, bins_assignment, positions, instance_name):
    num_bins = len(bins_assignment)
    
    if num_bins == 0:
        return

    # For stacking visualization, show bins stacked vertically
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'BPP_MS_S_C1 - {instance_name} - {num_bins} bins (Stacked)', fontsize=16)
    
    # Stack bins vertically
    for bin_idx, items_in_bin in enumerate(bins_assignment):
        bin_y_offset = bin_idx * bin_height
        
        # Draw bin boundary
        bin_rect = plt.Rectangle((0, bin_y_offset), bin_width, bin_height, 
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(bin_rect)
        
        # Add bin label
        ax.text(bin_width/2, bin_y_offset + bin_height/2, f'Bin {bin_idx + 1}', 
               ha='center', va='center', fontsize=12, weight='bold')
        
        # Draw rectangles in this bin
        for item_idx in items_in_bin:
            # Get dimensions (no rotation in this version)
            w, h = rectangles[item_idx][0], rectangles[item_idx][1]
            x, y = positions[item_idx]
            
            # Adjust y position for stacking
            rect = plt.Rectangle((x, y + bin_y_offset), w, h, 
                               edgecolor="#333", facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            
            # Add item label
            label = f"{item_idx + 1}"
            ax.text(x + w/2, y + bin_y_offset + h/2, label, 
                   ha='center', va='center', fontsize=10)
    
    # Set axis limits and labels
    ax.set_xlim(0, bin_width)
    ax.set_ylim(0, num_bins * bin_height)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height (Stacked)')
    ax.grid(True, alpha=0.3)
    
    # Add total height annotation
    ax.text(bin_width + 1, num_bins * bin_height / 2, 
           f'Total Height: {num_bins * bin_height}', 
           rotation=90, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'BPP_MS_S_C1/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def display_solution_each_bin(W, H, rectangles, positions, bins_used):
    """Display all bins in one window with subplots"""
    
    # Use the new colormap API for compatibility
    n_bins = len(bins_used)
    ncols = min(n_bins, 4)
    nrows = (n_bins + ncols - 1) // ncols
    plt.title(f'Solution for {instance_name}')

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    fig.suptitle(f'Solution for {instance_name}  - {n_bins} bins', fontsize=16)
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
        plt.savefig(f'BPP_MS_S_C1/{instance_name}_solution.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")
    
def read_file_instance(instance_name):
    """Read instance file based on instance name"""
    s = ''
    
    # Determine file path based on instance name
    if instance_name.startswith('BENG'):
        filepath = f"inputs/BENG/{instance_name}.txt"
    elif instance_name.startswith('CL_'):
        filepath = f"inputs/CLASS/{instance_name}.txt"
    else:
        # For other instances, try different folders
        possible_paths = [
            f"inputs/{instance_name}.txt",
            f"inputs/BENG/{instance_name}.txt",
            f"inputs/CLASS/{instance_name}.txt"
        ]
        
        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Could not find instance file for {instance_name}")
    
    try:
        with open(filepath, 'r') as f:
            s = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    return s.splitlines()

# Updated instance list with actual available instances
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

def calculate_lower_bound(bin_width, bin_height, rectangles):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = bin_width * bin_height
    return math.ceil(total_area / bin_area)

def first_fit_upper_bound(rectangles, W, H):
    """Finite First-Fit (FFF) upper bound for 2D bin packing without rotation."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        # For simplicity, try to place at bottom-left positions
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
        
        # Try to place in existing bins (no rotation in this version)
        for bin_rects in bins:
            if w <= W and h <= H:
                pos = fits(bin_rects, w, h, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], w, h))
                    placed = True
                    break
        
        if not placed:
            # Start a new bin, place at (0,0)
            if w <= W and h <= H:
                bins.append([(0, 0, w, h)])
            else:
                # Infeasible rectangle
                return float('inf')
    
    return len(bins)

def positive_range(end):
    if end < 0:
        return []
    return range(end)

# Save checkpoint for progress tracking
def save_checkpoint(instance_id, variables, clauses, bins, status="IN_PROGRESS"):
    checkpoint = {
        'Variables': variables,
        'Clauses': clauses,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }
    
    with open(f'checkpoint_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def BPP_MaxSat(W, H, lower_bound, upper_bound):
    """Incremental SAT-based BPP solver with stacking and Config 2 symmetry breaking"""
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot, bins_used
    global instance_name
    
    # MaxSAT encoding and solving (no incremental SAT, full MaxSAT encoding)
    import tempfile
    n_rects = len(rectangles)
    variables = {}
    counter = 1
    max_height = upper_bound * H
    min_height = lower_bound * H
    width = W
    # Create WCNF file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.wcnf') as file:
        wcnf_file = file.name
        file.write(f"c Bin Packing Problem MaxSAT, W={width}, H={H}, n={n_rects}\n")
        file.write(f"c Lower bound={lower_bound}, Upper bound={upper_bound}\n")
        # Variable encoding
        for i in range(n_rects):
            for j in range(n_rects):
                if i != j:
                    variables[f"lr{i + 1},{j + 1}"] = counter
                    counter += 1
                    variables[f"ud{i + 1},{j + 1}"] = counter
                    counter += 1
            for e in range(width):
                variables[f"px{i + 1},{e}"] = counter
                counter += 1
            for f in range(max_height):
                variables[f"py{i + 1},{f}"] = counter
                counter += 1
            
        # Height variables (ph_h: at least one rectangle at y >= h)
        for h in range(min_height, max_height + 1):
            variables[f"ph_{h}"] = counter
            counter += 1
        print(f"Total variables: {counter - 1}")
        hard_clauses = []
        # Order encoding
        for i in range(n_rects):
            for e in range(width - 1):
                hard_clauses.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
            for f in range(max_height - 1):
                hard_clauses.append([-variables[f"py{i + 1},{f}"], variables[f"py{i + 1},{f + 1}"]])
        # Height variable ordering
        for h in range(min_height, max_height):
            hard_clauses.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
        # Non-overlapping constraints (with rotation)
        max_width = max([int(rectangle[0]) for rectangle in rectangles])
        max_rect_height = max([int(rectangle[1]) for rectangle in rectangles if int(rectangle[0]) == max_width])
        def non_overlapping(i, j, h1, h2, v1, v2):
            i_width = rectangles[i][0]
            i_height = rectangles[i][1]
            j_width = rectangles[j][0]
            j_height = rectangles[j][1]

            if i_width + j_width > width:
                h1, h2 = False, False
            # Large-rectangles vertical
            elif i_height + j_height > max_height:
                v1, v2 = False, False
            # Same-sized rectangles
            elif rectangles[i] == rectangles[j]:
                h2 = False  # Only allow i left of j, not j left of i
            elif i_width == max_width and j_width > (W - i_width) / 2:
                h1 = False
            elif i_height == max_rect_height and j_height > (max_height - i_height) / 2:
                v1 = False


            # Four-literal disjunction for non-overlapping
            four_literal = []
            if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
            if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
            if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
            if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
            hard_clauses.append(four_literal)

            # Channeling constraints for left-right ordering
            if h1:
                for e in range(i_width):
                    if f"px{j + 1},{e}" in variables:
                        hard_clauses.append([-variables[f"lr{i + 1},{j + 1}"],
                                    -variables[f"px{j + 1},{e}"]])
            if h2:
                for e in range(j_width):
                    if f"px{i + 1},{e}" in variables:
                        hard_clauses.append([-variables[f"lr{j + 1},{i + 1}"],
                                    -variables[f"px{i + 1},{e}"]])
            
            # Channeling constraints for up-down ordering
            if v1:
                for f in range(i_height):
                    if f"py{j + 1},{f}" in variables:
                        hard_clauses.append([-variables[f"ud{i + 1},{j + 1}"],
                                    -variables[f"py{j + 1},{f}"]])
            if v2:
                for f in range(j_height):
                    if f"py{i + 1},{f}" in variables:
                        hard_clauses.append([-variables[f"ud{j + 1},{i + 1}"],
                                    -variables[f"py{i + 1},{f}"]])

            # No-overlap constraints for horizontal direction
            for e in positive_range(width - i_width):
                if h1 and f"px{j + 1},{e + i_width}" in variables:
                    hard_clauses.append([-variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"],
                                -variables[f"px{j + 1},{e + i_width}"]])
                if h2 and f"px{i + 1},{e + j_width}" in variables:
                    hard_clauses.append([-variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"],
                                -variables[f"px{i + 1},{e + j_width}"]])

            # No-overlap constraints for vertical direction
            for f in positive_range(max_height - i_height):
                if v1 and f"py{j + 1},{f + i_height}" in variables:
                    hard_clauses.append([-variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{f}"],
                                -variables[f"py{j + 1},{f + i_height}"]])
                if v2 and f"py{i + 1},{f + j_height}" in variables:
                    hard_clauses.append([-variables[f"ud{j + 1},{i + 1}"],
                                variables[f"py{j + 1},{f}"],
                                -variables[f"py{i + 1},{f + j_height}"]])

        max_width = max([int(rectangle[0]) for rectangle in rectangles])
        max_rect_height = max([int(rectangle[1]) for rectangle in rectangles if int(rectangle[0]) == max_width])
        for i in range(n_rects):
            for j in range(i + 1, n_rects):
                non_overlapping(i, j, True, True, True, True)  # Both rectangles in original orientation
        # Domain encoding
        for i in range(len(rectangles)):
        # Check if rectangle fits (no rotation allowed)
        # Horizontal domain constraint
            pos_x = width - rectangles[i][0]
            if pos_x >= 0:
                hard_clauses.append([variables[f"px{i + 1},{pos_x}"]])
        
            # Vertical domain constraint
            pos_y = max_height - rectangles[i][1]
            if pos_y >= 0:
                hard_clauses.append([variables[f"py{i + 1},{pos_y}"]])
        for k in range(1, upper_bound):
            for i in range(len(rectangles)):
                    h = rectangles[i][1]
                    hard_clauses.append([ variables[f"py{i + 1},{k * H - h}"],
                                -variables[f"py{i + 1},{k * H - 1}"]])
                    hard_clauses.append([-variables[f"py{i + 1},{k * H - h}"],
                                variables[f"py{i + 1},{k * H - 1}"]])
                    
                
        # Height-related constraints
        for h in range(min_height, max_height + 1):
            for i in range(n_rects):
                rect_height = rectangles[i][1]
                if h >= rect_height:
                    hard_clauses.append([-variables[f"ph_{h}"], variables[f"py{i + 1},{h - rect_height}"]])
        # Soft clauses: minimize height (ph_h)
        soft_clauses = []
        for h in range(min_height, max_height + 1, H):
            soft_clauses.append((1, [variables[f"ph_{h}"]]))
        # At least one ph_h true
        all_ph_vars = [variables[f"ph_{h}"] for h in range(min_height, max_height + 1, H)]
        hard_clauses.append(all_ph_vars)
        # Write hard clauses
        for clause in hard_clauses:
            file.write(f"h {' '.join(map(str, clause))} 0\n")
        
        # Write soft clauses with their weights
        for weight, clause in soft_clauses:
            file.write(f"{weight} {' '.join(map(str, clause))} 0\n")
        file.flush()
    variables_length = len(variables)
    clauses_length = len(hard_clauses) + len(soft_clauses)
    save_checkpoint(instance_id, variables_length, clauses_length, upper_bound, "IN_PROGRESS")
    # Call MaxSAT solver
    print(f"Running MaxSAT solver with {variables_length} variables and {len(hard_clauses)} hard clauses and {len(soft_clauses)} soft..")
    import subprocess
    try:
        print(f"Running tt-open-wbo-inc on {wcnf_file}...")
        result = subprocess.run(["./tt-open-wbo-inc-Glucose4_1_static", wcnf_file], capture_output=True, text=True)
        output = result.stdout
        # print(f"Solver output preview: {output}...")
        optimal_bins = upper_bound
        positions = [[0, 0] for _ in range(n_rects)]
        rotations = [False] * n_rects  # False means not rotated, True means rotated
        bins_used = [[] for _ in range(upper_bound)]
        print("Parsing solver output...")
        if "OPTIMUM FOUND" in output or "Optimal solution found" in output:
            print("Optimal solution found, parsing output...")
            for line in output.split('\n'):
                if line.startswith('v '):
                    binary_string = line[2:].strip()
                    print("\nSOLVER OUTPUT DIAGNOSTICS:")
                    print("=" * 50)

                    print(f"Characters in solution: {set(binary_string)}")
                    print(f"First 20 characters: {binary_string[:20]}")
                    print(f"Length of binary string: {len(binary_string)}")
                    print("=" * 50)
                    
                    true_vars = set()
                    
                    # Process the solution string
                    if " " in binary_string:
                        try:
                            for val in binary_string.split():
                                val_int = int(val)
                                if val_int > 0:  # Positive literals represent true variables
                                    true_vars.add(val_int)
                        except ValueError:
                            # Not integers, try as space-separated binary values
                            for i, val in enumerate(binary_string.split()):
                                if val == '1':
                                    true_vars.add(i + 1)  # 1-indexed
                    else:
                        # No spaces - treat as continuous binary string
                        for i, val in enumerate(binary_string):
                            if val == '1':
                                true_vars.add(i + 1)  # 1-indexed
                    print(f"True variables: {len(true_vars)}")
                    if not true_vars:
                        print("WARNING: Solution parsing failed. Using lower bound height as fallback.")
                        optimal_bins = lower_bound
                        best_height = lower_bound
                        
                        # Set default positions - simple greedy left-bottom placement
                        x_pos = 0
                        y_pos = 0
                        max_height = 0
                        for i in range(n_items):
                            # Default to non-rotated
                            w = rectangles[i][0]
                            h = rectangles[i][1]
                            
                            # If current rectangle doesn't fit in the current row, move to next row
                            if x_pos + w > width:
                                x_pos = 0
                                y_pos = max_height
                            
                            positions[i][0] = x_pos
                            positions[i][1] = y_pos
                    
                            
                            # Update position for next rectangle
                    else:
    
                        
                        # Extract positions
                        for i in range(n_items):
                            # Find x position (first position where px is true)
                            found_x = False
                            for e in range(width):
                                var_key = f"px{i + 1},{e}"
                                if var_key in variables and variables[var_key] in true_vars:
                                    if e == 0 or variables[f"px{i + 1},{e-1}"] not in true_vars:
                                        print(f"Found x-position {e} for rectangle {i}")
                                        positions[i][0] = e
                                        found_x = True
                                        break
                            if not found_x:
                                print(f"WARNING: Could not determine x-position for rectangle {i}!")
                            
                            # Find y position (first position where py is true)
                            found_y = False
                            for y_pos in range(max_height + 1):
                                var_key = f"py{i + 1},{y_pos}"
                                # Check if this y position is true
                                if var_key in variables and variables[var_key] in true_vars:
                                    if y_pos == 0 or variables[f"py{i + 1},{y_pos-1}"] not in true_vars:
                                        positions[i][1] = y_pos
                                        print(f"Found y-position {y_pos} for rectangle {i}")
                                        found_y = True
                                        break
                            if not found_y:
                                print(f"WARNING: Could not determine y-position for rectangle {i}!")
                        for i in range(len(positions)):
                            pos = positions[i]  
                            if len(pos) < 2:
                                print(f"WARNING: Position for rectangle {i} is incomplete: {pos}")
                            else:
                                bin_idx = pos[1] // H
                                pos[1] = pos[1] % H  # Ensure y position is within height bounds
                                bins_used[bin_idx].append(i)  # Store bin index (1-indexed)
                        #Remove empty bins
                        bins_used = [bin_items for bin_items in bins_used if bin_items]
                        optimal_bins = len(bins_used)
                    
        else:
            print("No optimal solution found within timeout.")
            print(f"Solver output: {output}")
        import os
        os.unlink(wcnf_file)
        return optimal_bins, (bins_used, positions)
    except Exception as e:
        print(f"Error running MaxSAT solver: {e}")

        import traceback
        traceback.print_exc()
        import os
        if os.path.exists(wcnf_file):
            os.unlink(wcnf_file)
        return None, None, None
if __name__ == "__main__":
    # Controller mode - running without arguments
    if len(sys.argv) == 1:
        # Create BPP_MS_S_C1 folder if it doesn't exist
        if not os.path.exists('BPP_MS_S_C1'):
            os.makedirs('BPP_MS_S_C1')
        
        # Read existing Excel file to check completed instances
        excel_file = 'BPP_MS_S_C1.xlsx'
        if os.path.exists(excel_file):
            existing_df = pd.read_excel(excel_file)
            completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
        else:
            existing_df = pd.DataFrame()
            completed_instances = []
        
        # Set timeout in seconds
        TIMEOUT = 900  # 10 minutes timeout
        
        # Run a subset of instances for testing
        test_instances = range(1, len(instances))  # Test all available instances
        
        for instance_id in test_instances:
            instance_name = instances[instance_id]
            
            # Check if instance already completed
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
            
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up any previous result file
            if os.path.exists(f'results_{instance_id}.json'):
                os.remove(f'results_{instance_id}.json')
            
            # Run the instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 BPP_MS_S_C1.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                
                time.sleep(1)
                
                result = None
                
                # Try to read results file first
                if os.path.exists(f'results_{instance_id}.json'):
                    with open(f'results_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                
                # If no results file, check checkpoint
                elif os.path.exists(f'checkpoint_{instance_id}.json'):
                    with open(f'checkpoint_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                    print(f"Instance {instance_name} timed out. Using checkpoint data.")
                
                # Process results if available
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Optimal Bins: {result['Optimal_Bins']}, Runtime: {result['Runtime']}")
                    
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
                            print(f"Error reading existing Excel file: {str(e)}")
                            existing_df = pd.DataFrame([result])
                    else:
                        existing_df = pd.DataFrame([result])
                    
                    existing_df.to_excel(excel_file, index=False)
                    print(f"Results saved to {excel_file}")
                        
                else:
                    print(f"No results or checkpoint found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up result files
            for file in [f'results_{instance_id}.json', f'checkpoint_{instance_id}.json']:
                if os.path.exists(file):
                    os.remove(file)
        
        print(f"\nAll test instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        instance_name = instances[instance_id]
        
        start = timeit.default_timer()
        
        try:
            print(f"\nProcessing instance {instance_name}")
            
            # Reset global best solution for this instance
            best_num_bins = float('inf')
            best_solution = None
            
            # Read input
            input_data = read_file_instance(instance_name)
            n_items = int(input_data[0])
            bin_size = input_data[1].split()
            bin_width = int(bin_size[0])
            bin_height = int(bin_size[1])
            rectangles = [[int(val) for val in line.split()] for line in input_data[2:2 + n_items]]
            
            # Calculate bounds
            lower_bound = calculate_lower_bound(bin_width, bin_height, rectangles)
            upper_bound = first_fit_upper_bound(rectangles, bin_width, bin_height)
            
            # Ensure upper bound is reasonable and at least lower bound
            if upper_bound == float('inf') or upper_bound > n_items:
                upper_bound = n_items
            
            if upper_bound < lower_bound:
                print(f"Warning: Upper bound ({upper_bound}) < Lower bound ({lower_bound}). Adjusting upper bound.")
                upper_bound = lower_bound
            
            print(f"Solving 2D Bin Packing with True OPP Formulation for instance {instance_name}")
            print(f"Bin size: {bin_width} x {bin_height}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            print(f"OPP Strip size: {bin_width} x {upper_bound * bin_height}")
            
            # Solve using Max-SAT with True OPP
            optimal_bins, solution = BPP_MaxSat(bin_width, bin_height, lower_bound, upper_bound)

            stop = timeit.default_timer()
            runtime = stop - start
            
            if optimal_bins:
                print(f"Optimal number of bins found: {optimal_bins}")
                print(f"OPP strip height used: {optimal_bins * bin_height}")
                
                # Display solution if found
                if solution:
                    bins_assignment, positions = solution
                    display_solution(bin_width, bin_height, rectangles, bins_assignment, positions, instance_name)
            final_bins = optimal_bins if optimal_bins != float('inf') else upper_bound

            # Create result
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': runtime,
                'Optimal_Bins': final_bins,
                'Status': 'COMPLETE'
            }
            
            # Save to Excel
            excel_file = 'BPP_MS_S_C1.xlsx'
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
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {final_bins}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()
            current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': timeit.default_timer() - start,
                'Optimal_Bins': current_bins,
                'Status': 'ERROR'
            }
            
            # Save error result to Excel
            excel_file = 'BPP_MS_S_C1.xlsx'
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