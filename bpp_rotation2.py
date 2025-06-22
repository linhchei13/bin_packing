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


def write_to_xlsx(result_dict):
    # Append the result to a list
    excel_results = []
    excel_results.append(result_dict)

    output_path = 'out/'

    # Write the results to an Excel file
    if not os.path.exists(output_path): os.makedirs(output_path)
    df = pd.DataFrame(excel_results)
    current_date = datetime.now().strftime('%Y-%m-%d')
    excel_file_path = f"{output_path}/results_{current_date}.xlsx"

    # Check if the file already exists
    if os.path.exists(excel_file_path):
        try:
            book = load_workbook(excel_file_path)
        except BadZipFile:
            book = Workbook()  # Create a new workbook if the file is not a valid Excel file

        # Check if the 'Results' sheet exists
        if 'Results' not in book.sheetnames:
            book.create_sheet('Results')  # Create 'Results' sheet if it doesn't exist

        sheet = book['Results']
        for row in dataframe_to_rows(df, index=False, header=False): sheet.append(row)
        book.save(excel_file_path)

    else: df.to_excel(excel_file_path, index=False, sheet_name='Results', header=False)

    print(f"Result added to Excel file: {os.path.abspath(excel_file_path)}\n")

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
            solver_time = format(time.time() - start, ".6f")
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
            bins_used = []
            for j in range(n_bins):
                bins_used.append([i for i in range(n_items) if result[f"B{i + 1},{j + 1}"] == True])
            timer.cancel()
            return ["sat", bins_used, pos, rotation, solver_time, counter, clauses_counter]
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
                for i in range(len(optimal_pos)):
                    optimal_pos[i].append(optimal_pos[i][1] // H) # append the bin number
                    optimal_pos[i][1] = optimal_pos[i][1] % H
                if lower_bound == upper_bound:
                    return -1
                else:
                    return BPP_binary_search(lower_bound, mid - 1)
            else:
                return BPP_binary_search(lower_bound, mid - 1)
    else:
        return -1

def export_to_xlsx(bpp_result, filepath, time):
    result_dict = {}
    if bpp_result is None:
        result_dict = {
            "Type": "SAT direct",
            "Data": filepath.split("/")[-1],
            "Number of items": n_items,
            "Bins": "-",  
            "Solver time": "-", 
            "Real time": time,
            "Variables": "-", 
            "Clauses": "-"}
        return
    else:
        bins = bpp_result[0]
        solver_time = bpp_result[3]
        num_variables = bpp_result[4]
        num_clauses = bpp_result[5]
        result_dict = {
            "Type": "SAT_direct",
            "Data": filepath.split("/")[-1],
            "Number of items": n_items,
            "Bins": len(bins),  
            "Solver time": solver_time, 
            "Real time": time,
            "Variables": num_variables, 
            "Clauses": num_clauses}
    write_to_xlsx(result_dict)

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


def print_solution(bpp_result):
    if bpp_result is None:
        print("No solution found")
        return
    else:
        bins = bpp_result[0]
        pos = bpp_result[1]
        rotation = bpp_result[2]
        solver_time = bpp_result[3]
        num_variables = bpp_result[4]
        num_clauses = bpp_result[5]
        for i in range(len(bins)):
            print("Bin", i + 1, "contains items", [(j + 1) for j in bins[i]])
            for j in bins[i]:
                if rotation[j]:
                    print("Rotated item", j + 1, rectangles[j], "at position", pos[j])
                else:
                    print("Item", j + 1, rectangles[j], "at position", pos[j])
            display_solution((W, H), [rectangles[j] for j in bins[i]], [pos[j] for j in bins[i]], [rotation[j] for j in bins[i]])
        print("--------------------")
        print("Solution found with", len(bins), "bins")
        print("Solver time:", solver_time)
        print("Number of variables:", num_variables)
        print("Number of clauses:", num_clauses)
 

def interrupt():
    print("Timeout")
    write_to_xlsx({
        "Type": "SAT direct",
        "Dataset": filepath.split("/")[-1],
        "Number of items": n_items,
        "Bins": "-",
        "Solver time": "timeout",
        "Real time": "timeout",
        "Number of variables": "-",
        "Number of clauses": "-"
    })
    os._exit(0)

if __name__ == '__main__':
    filepath = f"inputs/CLASS/CL_1_20_1.txt"
    lines = read_file(filepath)
    n_items = int(lines[0])
    W, H = map(int, lines[1].split())
    rectangles = [list(map(int, line.split())) for line in lines[2:n_items + 2]]
    lower_bound = math.ceil(sum([r[0] * r[1] for r in rectangles]) / (W * H))
    upper_bound = n_items
    print("Reading file", filepath.split("/")[-1])
    start = time.time()
    bpp_result = BPP_linear_search(lower_bound, upper_bound)
    
    stop = time.time()
    print("Time:", stop - start)
    export_to_xlsx(bpp_result, filepath, format(stop-start, ".3f"))
    print_solution(bpp_result)


