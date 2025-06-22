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
time_out = 200  
instance_name = ""
is_timeout = False

start = timeit.default_timer()
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
    print("Timeout")
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

def OPP(n_bins, W, H):
    # Define the variables
    global variables_length, clauses_length, best_bins, optimal_bins, optimal_pos, optimal_rot
    global upper_bound, is_timeout
    height = n_bins * H
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
    
    for k in range(1, n_bins):
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
                    save_checkpoint(instance_name, variables_length, clauses_length, best_bins)
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
                timer.cancel()  # Cancel the timer if we complete normally
                return ["sat", pos, rotation]
        except:
            timer.cancel()
            raise
            
def BPP_linear_search(lower, upper):
    global optimal_bins, optimal_pos, optimal_rot
    for n in range(lower, upper + 1):
        print(f"Trying with {n} bins")
        result = OPP(n, W, H)
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
    global is_timeout
    
    # Check timeout at the beginning
    if is_timeout or timeit.default_timer() - start > time_out:
        timeout_handler()
        
    if (lower <= upper):
        mid = (lower + upper) // 2
        print(f"Trying with {mid} bins")
        result = OPP(mid, W, H) 
        # Rest of the function stays the same
        if result[0] == "unsat":
            if lower == upper:
                return -1
            else:
                return BPP_binary_search(mid + 1, upper)
        else:
            if mid == lower:
                print(f"Optimal solution found with {mid} bins")
                global optimal_bins, optimal_pos, optimal_rot
                optimal_bins = mid
                optimal_pos = result[1]
                print(f"Optimal positions before: {optimal_pos}")
                optimal_rot = result[2]
                for i in range(len(optimal_pos)):
                    optimal_pos[i].append(optimal_pos[i][1] // H) # append the bin number
                    optimal_pos[i][1] = optimal_pos[i][1] % H
                if lower == upper:
                    return -1
                else:
                    return BPP_binary_search(lower, mid - 1)
            else:
                return BPP_binary_search(lower, mid - 1)
    else:
        return -1
    

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
    
    # Terminate the program
    os._exit(0)

# Modify your main execution code (at the bottom of the file)
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
        
        # Stop the timer if we complete normally
        timer.cancel()
        
    except Exception as e:
        timer.cancel()
        print(f"Error occurred: {str(e)}")
        raise e


