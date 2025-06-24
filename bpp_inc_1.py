import fileinput
from itertools import chain, combinations
import math
import os
import signal
from threading import Timer
import threading

import pandas as pd
from pysat.formula import CNF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib

from pysat.solvers import Glucose3
import datetime
import pandas as pd
import os
import sys
import time
from openpyxl import load_workbook
from openpyxl import Workbook
from zipfile import BadZipFile
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
import json
import timeit

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
time_out = 600  
instance_name = ""
is_timeout = False

start = 0
def read_file(file_path):
    global instance_name
    instance_name = file_path.split("/")[-1].split(".")[0]  # Lấy tên file không có phần mở rộng
    s = ""
    for line in fileinput.input(files=file_path):
        s += line
    return s.splitlines()

def positive_range(end):
    if (end < 0):
        return []
    return range(end)
def interupt(solver):
    print("Timeout reached, interrupting solver...")
    global is_timeout
    is_timeout = True
    solver.interrupt()

def display_solution(strip, rectangles, pos_circuits, rotation):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    ax = plt.gca()
    plt.title(strip)

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            rect = plt.Rectangle(pos_circuits[i],
                                 rectangles[i][0] if not rotation[i] else rectangles[i][1],
                                 rectangles[i][1] if not rotation[i] else rectangles[i][0],
                                 edgecolor="#333")
            ax.add_patch(rect)

    ax.set_xlim(0, strip[0])
    ax.set_ylim(0, strip[1] + 1)
    ax.set_xticks(range(strip[0] + 1))
    ax.set_yticks(range(strip[1] + 1))
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    # display plot
    plt.show()

def save_checkpoint(instance, variables, clauses, bins, status="IN_PROGRESS"):
    checkpoint = {
        'Variables': variables,
        'Clauses': clauses,
        'Runtime':  timeit.default_timer() - start,
        'Optimal_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }
    
    # Ghi ra file checkpoint
    with open(f'checkpoint_{instance}.json', 'w') as f:
        json.dump(checkpoint, f)

def BPP_incremental(W, H, lower_bound, upper_bound):
    # Define the variables
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot
    global is_timeout
    height = upper_bound * H
    width = W
    clauses = []
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
    for h in range(lower_bound * H, upper_bound * H):
        clauses.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
    
    # Add the 2-literal axiom clauses
    for i in range(len(rectangles)):
        for e in range(width - 1):  # -1 because we're using e+1 in the clause
            clauses.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        for f in range(height - 1):  # -1 because we're using f+1 in the clause
            clauses.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])
    # Add non-overlapping constraints

    def non_overlapping(rotated, i, j, h1, h2, v1, v2):
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

        # Square symmertry breaking, if i is square than it cannot be rotated
        if i_width == i_height and rotated:
            i_square = True
            clauses.append([-variables[f"r{i + 1}"]])
        else:
            i_square = False

        if j_width == j_height and rotated:
            j_square = True
            clauses.append([-variables[f"r{j + 1}"]])
        else:
            j_square = False

        # lri,j v lrj,i v udi,j v udj,i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])

        clauses.append(four_literal + [i_rotation])
        clauses.append(four_literal + [j_rotation])

        #¬lri, j ∨ ¬pxj, e
        if h1 and not i_square:
            for e in range(min(width, i_width)):
                    clauses.append([i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                -variables[f"px{j + 1},{e}"]])
        # ¬lrj,i ∨ ¬pxi,e
        if h2 and not j_square:
            for e in range(min(width, j_width)):
                    clauses.append([j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                -variables[f"px{i + 1},{e}"]])
        # ¬udi,j ∨ ¬pyj,f
        if v1 and not i_square:
            for f in range(min(height, i_height)):
                    clauses.append([i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                -variables[f"py{j + 1},{f}"]])
        #¬udj, i ∨ ¬pyi, f,
        if v2 and not j_square:
            for f in range(min(height, j_height)):
                    clauses.append([j_rotation,
                                -variables[f"ud{j + 1},{i + 1}"],
                                -variables[f"py{i + 1},{f}"]])

        for e in positive_range(width - i_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1 and not i_square:
                    clauses.append([i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"],
                                -variables[f"px{j + 1},{e + i_width}"]])

        for e in positive_range(width - j_width):
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2 and not j_square:
                    clauses.append([j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"],
                                -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(height - i_height):
            # udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1 and not i_square:
                    clauses.append([i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{f}"],
                                -variables[f"py{j + 1},{f + i_height}"]])
        for f in positive_range(height - j_height):
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2 and not j_square:
                    clauses.append([j_rotation,
                                -variables[f"ud{j + 1},{i + 1}"],
                                variables[f"py{j + 1},{f}"],
                                -variables[f"py{i + 1},{f + j_height}"]])

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
           #  #Large-rectangles horizontal
            if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > width:
                non_overlapping(False, i, j, False, False, True, True)
                non_overlapping(True, i, j, False, False, True, True)
            # Large rectangles vertical
            elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > height:
                non_overlapping(False, i, j, True, True, False, False)
                non_overlapping(True, i, j, True, True, False, False)

            # Same rectangle and is a square
            elif rectangles[i] == rectangles[j]:
                if rectangles[i][0] == rectangles[i][1]:
                    clauses.append([-variables[f"r{i + 1}"]])
                    clauses.append([-variables[f"r{j + 1}"]])
                    non_overlapping(False,i ,j, True, False, True, True)
                else:
                    non_overlapping(False,i ,j, True, False, True, True)
                    non_overlapping(True,i ,j, True, False, True, True)
           # #normal rectangles
            else:
                non_overlapping(False, i, j, True, True, True, True)
                non_overlapping(True, i, j, True, True, True, True)

    # # Domain encoding to ensure every rectangle stays inside strip's boundary
    for i in range(len(rectangles)):
            if rectangles[i][0] > width: #if rectangle[i]'s width larger than strip's width, it has to be rotated
                clauses.append([variables[f"r{i + 1}"]])
            else:
                clauses.append([variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][0]}"]])
       
            if rectangles[i][1] > height:
                clauses.append([variables[f"r{i + 1}"]])
            else:
                clauses.append([variables[f"r{i + 1}"],
                            variables[f"py{i + 1},{height - rectangles[i][1]}"]])

            # Rotated
            if rectangles[i][1] > width:
                clauses.append([-variables[f"r{i + 1}"]])
            else:
                clauses.append([-variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][1]}"]])
            if rectangles[i][0] > height:
                clauses.append([-variables[f"r{i + 1}"]])
            else:
                clauses.append([-variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{height - rectangles[i][0]}"]])
    
    for k in range(1, upper_bound + 1):
         for i in range(len(rectangles)):
            # Not rotated
            h = rectangles[i][1]
            clauses.append([variables[f"r{i + 1}"], variables[f"py{i + 1},{k * H - h}"],
                        -variables[f"py{i + 1},{k * H - 1}"]])
            clauses.append([variables[f"r{i + 1}"], -variables[f"py{i + 1},{k * H - h}"],
                        variables[f"py{i + 1},{k * H - 1}"]])
            
            # Rotated
            h = rectangles[i][0]
            clauses.append([-variables[f"r{i + 1}"], variables[f"py{i + 1},{k * H - h}"],
                        -variables[f"py{i + 1},{k * H - 1}"]])
            clauses.append([-variables[f"r{i + 1}"], -variables[f"py{i + 1},{k * H - h}"],
                        variables[f"py{i + 1},{k * H - 1}"]])
            
    for h in range(lower_bound* H, upper_bound*H + 1):
        for i in range(len(rectangles)):
            # Normal orientation
            rect_height = rectangles[i][1]
            if h >= rect_height:
                clauses.append([-variables[f"ph_{h}"], variables[f"r{i+1}"], 
                          variables[f"py{i+1},{h - rect_height}"]])
            
            # Rotated orientation
            rotated_height = rectangles[i][0]
            if h >= rotated_height:
                clauses.append([-variables[f"ph_{h}"], -variables[f"r{i+1}"], 
                          variables[f"py{i+1},{h - rotated_height}"]])
    stop = time.time()
    print("Encoding Time:", format(stop-start, ".6f"))  
    variables_length = len(variables)
    clauses_length = len(clauses)
    save_checkpoint(instance_name, variables_length, clauses_length, best_bins, "IN_PROGRESS")      
    with Glucose3(use_timer=True) as solver:
        # Add the clauses to the solver
        for clause in clauses:
            solver.add_clause(clause)
        timer = Timer(time_out, interupt, [solver])
        timer.start()    

        best_model = None
        
        # Binary search with incremental solving
        current_lb = lower_bound
        current_ub = upper_bound
        
        while current_lb <= current_ub:
            if timeit.default_timer() - start > time_out:
                print(f"Timeout reached after {time_out} seconds. Saving current best solution.")
                timeout_handler()
                break
            # Check timeout before each iteration
            mid = (current_lb + current_ub) // 2
            print(f"Trying height: {mid*H} (lower={current_lb}, upper={current_ub})")
            
            # Set up assumptions for this iteration - test if we can pack with height ≤ mid
            assumptions = [variables[f"ph_{mid*H}"]]
            
            # Lưu checkpoint trước khi giải
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
                
                # Save the model for reuse
                best_model = solver.get_model()
                
                # Extract positions and rotations from the model
                positions = [[0, 0] for _ in range(n_items)]
                rotations = [False for _ in range(n_items)]
                
                # Convert model to dictionary for faster lookup
                true_vars = set(var for var in best_model if var > 0)
                
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
                
                # Save the best positions and rotations
                best_positions = positions
                best_rotations = rotations
                
                # Update search range - try lower height
                current_ub = mid - 1
            
            else:
                # No solution with height ≤ mid
                # Update search range - try higher height
                print(f"No solution with height ≤ {mid*H}, trying higher")
                current_lb = mid + 1

        variables_length = len(variables)
        clauses_length = len(clauses)

        # Final validation of the solution
        if positions is None:
            return None, None, None
        
        print(f"Final optimal height: {optimal_bins}")
        timer.cancel()  # Stop the timer
        return optimal_bins, positions, rotations

# Add/modify the timeout handler function
def timeout_handler():
    global best_bins, upper_bound, is_timeout
    is_timeout = True
    print(f"\nTimeout reached after {time_out} seconds. Saving current best solution.")
    
    # Get the best height found so far
    current_bins = best_bins if best_bins != float('inf') else upper_bound
    print(f"Best solution found before timeout: {current_bins} bins")
    
    # Save result as JSON
    result = {
        'Instance': instance_name,
        'Variables': variables_length,
        'Clauses': clauses_length,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': current_bins,
        'Status': 'TIMEOUT',
    }
    
    with open(f'results_{instance_name}.json', 'w') as f:
        json.dump(result, f)
    write_to_xlsx(result)
    # Terminate the program
    os._exit(0)

def write_to_xlsx(result_dict):
    df = pd.DataFrame([result_dict])
    output_path = 'bpp_inc.xlsx'
    # If file exists, append, else create new
    if os.path.exists(output_path):
        old_df = pd.read_excel(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_excel(output_path, index=False)

def display_solution_each_bin(W, H, rectangles, positions, rotations):
    # Group rectangles by bin
    bins = {}
    for i, pos in enumerate(positions):
        bin_id = pos[2] if len(pos) > 2 else 0
        if bin_id not in bins:
            bins[bin_id] = []
        bins[bin_id].append((i, pos, rotations[i]))
    # Use the new colormap API for compatibility
    cmap = matplotlib.colormaps.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(rectangles))]
    n_bins = len(bins)
    ncols = min(n_bins, 4)
    nrows = (n_bins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if n_bins == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    axes = axes.flatten()
    for idx, (bin_id, rects) in enumerate(sorted(bins.items())):
        ax = axes[idx]
        ax.set_title(f'Bin {bin_id+1}')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        for ridx, pos, rot in rects:
            w, h = rectangles[ridx]
            if rot:
                w, h = h, w
            x0, y0 = pos[0], pos[1]
            rect_patch = plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=colors[ridx], alpha=0.7)
            ax.add_patch(rect_patch)
            cx, cy = x0 + w/2, y0 + h/2
            rgb = matplotlib.colors.to_rgb(colors[ridx])
            brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            text_color = 'white' if brightness < 0.6 else 'black'
            rot_info = 'R' if rot else 'NR'
            ax.text(cx, cy, f'{ridx+1}\n{rot_info}', ha='center', va='center', color=text_color, fontweight='bold')
        ax.set_xticks(range(0, W+1, max(1, W//10)))
        ax.set_yticks(range(0, H+1, max(1, H//10)))
        ax.grid(True, linestyle='--', alpha=0.3)
    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for i in range(1, 11):
        timer = threading.Timer(time_out, timeout_handler)
        timer.daemon = True
        timer.start()
        try:
            start = timeit.default_timer()
            input_data = read_file(f"inputs/class/CL_1_80_{i}.txt")
            n_items = int(input_data[0])
            W, H = map(int, input_data[1].split())
            rectangles = [[int(val) for val in i.split()] for i in input_data[2:]]
            print(f"Input file: {instance_name}")
            print(f"Number of items: {n_items}, Width: {W}, Height: {H}")
            print(f"Rectangles: {rectangles}")
            total_area = sum([w * h for w, h in rectangles])
            lower_bound = math.ceil(total_area / (W * H))
            print(f"Lower bound: {lower_bound}")
            upper_bound = n_items
            best_bins = upper_bound
            optimal_bins, optimal_pos, optimal_rot= BPP_incremental(W, H, lower_bound, upper_bound)
            
            # Save result to Excel after successful run
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': timeit.default_timer() - start,
                'Optimal_Bins': optimal_bins,
                'Status': 'SUCCESS',
            }
            write_to_xlsx(result)
            print(f"Optimal bins: {optimal_bins}")
            print(f"Optimal positions: {optimal_pos}")
            print(f"Optimal rotations: {optimal_rot}")
            timer.cancel()
            # Display solution for each bin
            # display_solution_each_bin(W, H, rectangles, optimal_pos, optimal_rot)
            
        except Exception as e:
            timer.cancel()
            print(f"Error occurred: {str(e)}")
            raise e