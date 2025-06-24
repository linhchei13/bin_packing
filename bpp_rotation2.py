from itertools import chain, combinations
import json
import math
from threading import Timer

from pysat.formula import CNF
from pysat.solvers import Solver

import matplotlib.pyplot as plt
import timeit
from typing import List
from pysat.solvers import Glucose3, Solver
from threading import Timer

import pandas as pd
import os
import sys
import time
from openpyxl import load_workbook
from openpyxl import Workbook
from zipfile import BadZipFile
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
import fileinput
import json
import matplotlib
import numpy as np
import threading


class TimeoutException(Exception): pass
n_items = 0
W, H = 0, 0
upper_bound = 0
rectangles = []
variables_length = 0
clauses_length = 0
best_bins = 0
best_pos = []
best_rot = []
optimal_bins = 0
optimal_pos = []
optimal_rot = []
time_out = 200  
instance_name = ""

start = timeit.default_timer()
#read file
def read_file(file_path):
    global instance_name
    instance_name = file_path.split("/")[-1].split(".")[0]  # Lấy tên file không có phần mở rộng
    s = ""
    for line in fileinput.input(files=file_path):
        s += line
    return s.splitlines()

def interrupt(solver):
    solver.interrupt()
def positive_range(end):
    if (end < 0):
        return []
    return range(end)

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


def BPP(n_bins):
# Define the variables
    start = time.time()
    height = H
    width = W
    cnf = CNF()
    variables = {}
    counter = 1
    global clauses_counter
    clauses_counter = 0

    # B_i_j = 1 if item i is placed in bin j
    for i in range(n_items):
        for j in range(n_bins):
            variables[f"B{i + 1},{j + 1}"] = counter 
            counter += 1

    for i in range(len(rectangles)):
        for i2 in range(len(rectangles)):
            if i != i2:
                variables[f"lr{i + 1},{i2 + 1}"] = counter  # lri,rj
                counter += 1
                variables[f"ud{i + 1},{i2 + 1}"] = counter  # uri,rj
                counter += 1
        for e in range(width):
            variables[f"px{i + 1},{e}"] = counter  # pxi,e
            counter += 1
        for f in range(height):
            variables[f"py{i + 1},{f}"] = counter  # pyi,f
            counter += 1

    # Rotated variables
    for i in range(len(rectangles)):
        variables[f"r{i + 1}"] = counter
        counter += 1

    # Exactly one bin for each item
    for i in range(n_items):
        cnf.append([variables[f"B{i + 1},{j + 1}"] for j in range(n_bins)])
        for j1 in range(n_bins):
            for j2 in range(j1 + 1, n_bins):
                cnf.append([-variables[f"B{i + 1},{j1 + 1}"], -variables[f"B{i + 1},{j2 + 1}"]])


    # Order constraints 
    for i in range(len(rectangles)):
        for e in range(width - 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
            
        for f in range(height - 1):  # -1 because we're using f+1 in the clause
            cnf.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])
            
    # Add non-overlapping constraints
    def non_overlapping(rotated, i1, i2, h1, h2, v1, v2, j):
        if not rotated:
            i1_width = rectangles[i1][0]
            i1_height = rectangles[i1][1]
            i2_width = rectangles[i2][0]
            i2_height = rectangles[i2][1]
            i1_rotation = variables[f"r{i1 + 1}"]
            i2_rotation = variables[f"r{i2 + 1}"]
        else:
            i1_width = rectangles[i1][1]
            i1_height = rectangles[i1][0]
            i2_width = rectangles[i2][1]
            i2_height = rectangles[i2][0]
            i1_rotation = -variables[f"r{i1 + 1}"]
            i2_rotation = -variables[f"r{i2 + 1}"]
            
        # if rectangle i1, i2 are in the same bin j, then they cannot overlap 
        bin_constraint = [-variables[f"B{i1 + 1},{j + 1}"], -variables[f"B{i2 + 1},{j + 1}"]]
        # Square symmertry breaking, if i is square than it cannot be rotated
        if i1_width == i1_height and rotated:
            cnf.append(bin_constraint+[-variables[f"r{i1 + 1}"]])
            i1_square = True
        else:
            i1_square = False

        if i2_width == i2_height and rotated:
            cnf.append(bin_constraint+[-variables[f"r{i2 + 1}"]])
            i2_square = True
        else:
            i2_square = False
        
        # lri,j v lrj,i v udi,j v udj,i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i1 + 1},{i2 + 1}"])
        if h2: four_literal.append(variables[f"lr{i2 + 1},{i1 + 1}"])
        if v1: four_literal.append(variables[f"ud{i1 + 1},{i2 + 1}"])
        if v2: four_literal.append(variables[f"ud{i2 + 1},{i1 + 1}"])

        cnf.append(four_literal + [i1_rotation] + bin_constraint)
        cnf.append(four_literal + [i2_rotation] + bin_constraint)

        # ¬lri, j ∨ ¬pxj, e
        if h1 and not i1_square:
            for e in range(min(width, i1_width)):
                    cnf.append(bin_constraint + [i1_rotation,
                                -variables[f"lr{i1 + 1},{i2 + 1}"],
                                -variables[f"px{i2 + 1},{e}"]])
        # ¬lrj,i ∨ ¬pxi,e
        if h2 and not i2_square:
            for e in range(min(width, i2_width)):
                    cnf.append(bin_constraint + [i2_rotation,
                                -variables[f"lr{i2 + 1},{i1 + 1}"],
                                -variables[f"px{i1 + 1},{e}"]])
                    # ¬udi,j ∨ ¬pyj,f
        if v1  and not i1_square:
            for f in range(min(height, i1_height)):
                cnf.append(bin_constraint + [i1_rotation,
                            -variables[f"ud{i1 + 1},{i2 + 1}"],
                            -variables[f"py{i2 + 1},{f}"]])
        # ¬udj, i ∨ ¬pyi, f,
        if v2 and not i2_square:
            for f in range(min(height, i2_height)):
                    cnf.append(bin_constraint + [i2_rotation,
                                -variables[f"ud{i2 + 1},{i1 + 1}"],
                                -variables[f"py{i1 + 1},{f}"]])

        for e in positive_range(width - i1_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1 and not i1_square:
                    cnf.append(bin_constraint + [i1_rotation,
                                -variables[f"lr{i1 + 1},{i2 + 1}"],
                                variables[f"px{i1 + 1},{e}"],
                                -variables[f"px{i2 + 1},{e + i1_width}"]])

        for e in positive_range(width - i2_width):
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2 and not i2_square:
                cnf.append(bin_constraint + [i2_rotation,
                                    -variables[f"lr{i2 + 1},{i1 + 1}"],
                                    variables[f"px{i2 + 1},{e}"],
                                    -variables[f"px{i1 + 1},{e + i2_width}"]])

        for f in positive_range(height - i1_height):
            # udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1 and not i1_square:
                    cnf.append(bin_constraint + [i1_rotation,
                                -variables[f"ud{i1 + 1},{i2 + 1}"],
                                variables[f"py{i1 + 1},{f}"],
                                -variables[f"py{i2 + 1},{f + i1_height}"]])
                    
        for f in positive_range(height - i2_height):
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2 and not i2_square:
                    cnf.append(bin_constraint + [i2_rotation,
                                -variables[f"ud{i2 + 1},{i1 + 1}"],
                                variables[f"py{i2 + 1},{f}"],
                                -variables[f"py{i1 + 1},{f + i2_height}"]])

    for j in range(n_bins):
        for i in range(len(rectangles)):
            for i2 in range(i + 1, len(rectangles)):
                #Large-rectangles horizontal
                if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[i2][0], rectangles[i2][1]) > width:
                    non_overlapping(False, i, i2, False, False, True, True, j)
                    non_overlapping(True, i, i2, False, False, True, True, j)
                # Large rectangles vertical
                elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[i2][0], rectangles[i2][1]) > height:
                    non_overlapping(False, i, i2, True, True, False, False, j)
                    non_overlapping(True, i, i2, True, True, False, False, j)

                # Same rectangle and is a square
                elif rectangles[i] == rectangles[i2]:
                    if rectangles[i][0] == rectangles[i][1]:
                        cnf.append([-variables[f"r{i + 1}"]])
                        cnf.append([-variables[f"r{i2 + 1}"]])
                        non_overlapping(False,i ,i2, True, False, True, True, j)
                    else:
                        non_overlapping(False,i ,i2, True, False, True, True, j)
                        non_overlapping(True,i ,i2, True, False, True, True, j)
            # #normal rectangles
                else:
                    non_overlapping(False, i, i2, True, True, True, True, j)
                    non_overlapping(True, i, i2, True, True, True, True, j)
    # Domain encoding to ensure every rectangle stays inside strip's boundary
    for j in range(n_bins):
        for i in range(len(rectangles)):
            if rectangles[i][0] > width:
                cnf.append([-variables[f"B{i+1},{j+1}"], variables[f"r{i + 1}"]])
            else:
                cnf.append([-variables[f"B{i+1},{j+1}"], variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][0]}"]])
        
            if rectangles[i][1] > height:
                cnf.append([-variables[f"B{i+1},{j+1}"], variables[f"r{i + 1}"]])
            
            else:    
                cnf.append([-variables[f"B{i+1},{j+1}"], variables[f"r{i + 1}"],
                            variables[f"py{i + 1},{height - rectangles[i][1]}"]])
                

            # Rotated
            if rectangles[i][1] > width:
                cnf.append([-variables[f"B{i+1},{j+1}"],-variables[f"r{i + 1}"]])
            
            else:
                cnf.append([-variables[f"B{i+1},{j+1}"],-variables[f"r{i + 1}"],
                                    variables[f"px{i + 1},{width - rectangles[i][1]}"]])
        
            if rectangles[i][0] > height:
                cnf.append([-variables[f"B{i+1},{j+1}"],-variables[f"r{i + 1}"]])    
               
            else:
                cnf.append([-variables[f"B{i+1},{j+1}"],-variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{height - rectangles[i][0]}"]])
    print("Encoding time:", time.time() - start)
    save_checkpoint(instance_name, len(variables), len(cnf.clauses), n_bins)
    solver = Glucose3(use_timer=True)
    sat_status = False
    
    solver.append_formula(cnf)
    timer = Timer(100, interrupt, [solver])
    timer.start()
    start = time.time()
    try: 
        print("Solving")
        sat_status = solver.solve_limited(expect_interrupt=True)
        if sat_status == None:
            timer.cancel()
            print("timeout")
            return "timeout", None, None, None,None, len(variables), len(cnf.clauses)
        elif sat_status == False:
            timer.cancel()
            print("unsat")
            return("unsat")
        else:
            pos = [[0 for i in range(2)] for j in range(len(rectangles))]
            rotation = []
            model = solver.get_model()
            print("SAT")
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
                for j in range(n_bins):
                    if result[f"B{i + 1},{j + 1}"] == True:
                        pos[i].append(j)
            bins_used = [[] for _ in range(n_bins)]
            for i in range(len(rectangles)):
                bins_used[pos[i][2]].append(i)
            timer.cancel()
            return ["sat", pos, rotation]
    except TimeoutException:
        timer.cancel()
        print("Timeout")
        return("timeout")
  
def BPP_linear_search(lower_bound, upper_bound):
    global optimal_bins, optimal_pos, optimal_rot, best_bins, best_pos, best_rot
    result = None
    for k in range(lower_bound, upper_bound):
        print(f"Trying with {k} bins")
        try:
            result = BPP(k)
            if result[0] == "sat":
                print(f"Solution found with {k} bins")
                optimal_bins = k
                optimal_pos = result[1]
                optimal_rot = result[2]
                return result[1:]
            else:
                print("No solution found")
                result = result[1:]
        except TimeoutException:
            print("Time out")
    return result

def BPP_binary_search(lower_bound, upper_bound):
    global optimal_bins, optimal_pos, optimal_rot, best_bins, best_pos, best_rot
    result = None
    if (lower_bound <= upper_bound):
        mid = (lower_bound + upper_bound) // 2
        print(f"Trying with {mid} bins")
        result = BPP(mid) 
        # Rest of the function stays the same
        if result[0] == "unsat":
            if lower_bound == upper_bound:
                return -1
            else:
                return BPP_binary_search(mid + 1, upper_bound)
        else:
            if mid == lower_bound:
                print(f"Optimal solution found with {mid} bins")
                global optimal_bins, optimal_pos, optimal_rot
                optimal_bins = mid
                optimal_pos = result[1]
                print(f"Optimal positions before: {optimal_pos}")
                optimal_rot = result[2]
                if lower_bound == upper_bound:
                    return -1
                else:
                    return BPP_binary_search(lower_bound, mid - 1)
            else:
                return BPP_binary_search(lower_bound, mid - 1)
    else:
        return -1

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

def display_solution(strip, rectangles, pos_circuits, rotation):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    ax = plt.gca()
    plt.title(strip)
    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            width = rectangles[i][0] if not rotation[i] else rectangles[i][1]
            height = rectangles[i][1] if not rotation[i] else rectangles[i][0]
            rect = plt.Rectangle(pos_circuits[i], width, height, 
                                edgecolor="#333")
            ax.add_patch(rect)
            # Add item label at the center of the rectangle
            rx, ry = pos_circuits[i]
            ax.text(rx + width/2, ry + height/2, str(i+1), 
                   ha='center', va='center', fontsize=9, color='black')

    ax.set_xlim(0, strip[0])
    ax.set_ylim(0, strip[1] + 1)
    ax.set_xticks(range(strip[0] + 1))
    ax.set_yticks(range(strip[1] + 1))
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    # display plot
    plt.show()


def write_to_xlsx(result_dict):
    df = pd.DataFrame([result_dict])
    output_path = 'bpp_rotation2.xlsx'
    # If file exists, append, else create new
    if os.path.exists(output_path):
        old_df = pd.read_excel(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_excel(output_path, index=False)

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

if __name__ == "__main__":
    # Set up the timeout timer
    timer = threading.Timer(time_out, timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        input_data = read_file("inputs/class/CL_1_60_1.txt")
        n_items = int(input_data[0])
        W, H = map(int, input_data[1].split())
        rectangles = [[int(val) for val in i.split()] for i in input_data[2:]]
        print(f"Number of items: {n_items}, Width: {W}, Height: {H}")
        print(f"Rectangles: {rectangles}")
        total_area = sum([w * h for w, h in rectangles])
        lower_bound = math.ceil(total_area / (W * H))
        print(f"Lower bound: {lower_bound}")
        upper_bound = n_items
        best_bins = upper_bound
        BPP_binary_search(lower_bound, upper_bound)
        print(f"Optimal bins: {optimal_bins}")
        print(f"Optimal positions: {optimal_pos}")
        print(f"Optimal rotations: {optimal_rot}")
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
        display_solution_each_bin(W, H, rectangles, optimal_pos, optimal_rot)
        # Stop the timer if we complete normally
        timer.cancel()
        
    except Exception as e:
        timer.cancel()
        print(f"Error occurred: {str(e)}")
        raise e


