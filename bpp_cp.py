from ortools.sat.python import cp_model
import sys
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

def read_input(file_path):
    with open(file_path) as f:
        data = f.readlines()
        n_packs = int(data[0])
        n_bins = n_packs
        W, H = map(int, data[1].split())
        packs = []
        for i in range(2, n_packs+2):
            packs.append(tuple(map(int, data[i].split())))

    return n_packs, n_bins, packs, W, H

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
    
def BPP_CP(file_path, time_limit):
    n_items, n_bins, items, W, H = read_input(file_path)
    max_pack_width = max(x[0] for x in items)
    max_pack_height = max(x[1] for x in items)
    print(f"Max pack width: {max_pack_width}")
    start = time.time()
    # Creates the model
    model = cp_model.CpModel()

    # 
    # Variables
    # s
    X = {}
    R = []
    for i in range(n_items):
        # R[i] = 1 iff item i is rotated
        R.append(model.NewBoolVar(f'package_{i}_rotated'))
        for j in range(n_bins):
            # X[i, j] = 1 iff item i is packed in bin j.
            X[i, j] = model.NewBoolVar(f'pack_{i}_in_bin_{j}')

    # Z[j] = 1 iff bin j has been used.
    Z = [model.NewBoolVar(f'bin_{j}_is_used)') for j in range(n_bins)]

    # Width and height of each pack
    width = []
    height = []
    # top right corner coordinate 
    x = []
    y = [] 
    for i in range(n_items):
        width.append(model.NewIntVar(0, max_pack_width, f'width_{i}'))
        height.append(model.NewIntVar(0, max_pack_height, f'height_{i}'))

        x.append(model.NewIntVar(0, W, f'x_{i}'))
        y.append(model.NewIntVar(0, H, f'y_{i}'))

        # if pack rotated -> switch the height and width
        model.Add(width[i] == items[i][0]).OnlyEnforceIf(R[i].Not())
        model.Add(width[i] == items[i][1]).OnlyEnforceIf(R[i])
        model.Add(height[i] == items[i][1]).OnlyEnforceIf(R[i].Not())
        model.Add(height[i] == items[i][0]).OnlyEnforceIf(R[i])
    # 
    # Constraint
    # 
    # Each pack can only be placed in one bin
    for i in range(n_items):
        model.Add(sum(X[i, j] for j in range(n_bins)) == 1)
        
    # if pack in bin, it cannot exceed the bin size
    for i in range(n_items):
        for j in range(n_bins):
            model.Add(x[i] <= W-width[i]).OnlyEnforceIf(X[i, j])
            model.Add(y[i] <= H-height[i]).OnlyEnforceIf(X[i, j])
            model.Add(x[i] >= 0).OnlyEnforceIf(X[i, j])
            model.Add(y[i] >= 0).OnlyEnforceIf(X[i, j])            

    # If 2 pack in the same bin they cannot overlap
    for i in range(n_items-1):
        for k in range(i+1, n_items):
            a1 = model.NewBoolVar('a1')        
            model.Add(x[i] <= x[k] - width[i]).OnlyEnforceIf(a1)
            model.Add(x[i] > x[k] - width[i]).OnlyEnforceIf(a1.Not())
            a2 = model.NewBoolVar('a2')        
            model.Add(y[i] <= y[k] - height[i]).OnlyEnforceIf(a2)
            model.Add(y[i] > y[k] - height[i]).OnlyEnforceIf(a2.Not())
            a3 = model.NewBoolVar('a3')        
            model.Add(x[k] <= x[i] - width[k]).OnlyEnforceIf(a3)
            model.Add(x[k] > x[i] - width[k]).OnlyEnforceIf(a3.Not())
            a4 = model.NewBoolVar('a4')        
            model.Add(y[k] <= y[i] - height[k]).OnlyEnforceIf(a4)
            model.Add(y[k] > y[i] - height[k]).OnlyEnforceIf(a4.Not())

            for j in range(n_bins):
                model.AddBoolOr(a1, a2, a3, a4).OnlyEnforceIf(X[i, j], X[k, j])

    # Find which bin has been used               
    for j in range(n_bins):
        b1 = model.NewBoolVar('b')
        model.Add(sum(X[i, j] for i in range(n_items)) == 0).OnlyEnforceIf(b1)
        model.Add(Z[j] == 0).OnlyEnforceIf(b1)
        model.Add(sum(X[i, j] for i in range(n_items)) != 0).OnlyEnforceIf(b1.Not())
        model.Add(Z[j] == 1).OnlyEnforceIf(b1.Not())

    # Objective function
    cost = sum(Z[j] for j in range(n_bins))
    model.Minimize(cost)
    result_dict = {}
    # Creates a solver and solves the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    
    # Print the results
    print('----------------Given data----------------')
    print(f'Number of pack given: {n_items}')
    print(f'Number of bin given : {n_bins}')
    stop = time.time()
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('--------------Solution Found--------------')

        # Uncomment if you want to see the way to put packages in bins
        # Not necessary in the statistics, so we comment it out
        # for i in range(n_items):
        #     if solver.Value(R[i]) == 1:
        #         print(f'Rotate pack {i+1} and put', end=' ')
        #     else:
        #         print(f'Put pack {i+1}', end=' ')
        #     for j in range(n_bins):
        #         if solver.Value(X[i, j]) == 1:
        #             print(f'in bin {j+1}', end='/n')
        # print(f'Number of bin used  : {sum(solver.Value(Z[i]) for i in range(n_bins))}')
        # print('----------------Statistics----------------')
        print(f'Status              : {solver.StatusName(status)}')
        print(f'Time limit          : {time_limit}')
        print(f'Running time        : {solver.UserTime()}')
        print(f'Explored branches   : {solver.NumBranches()}')
    
        result_dict = {
            "Type": "Ortools CP",
            "Data": file_path.split("/")[-1],
            "Number of items": n_items,
            "Bins": sum(solver.Value(Z[i]) for i in range(n_bins)),  
            "Solver time": format(solver.UserTime(), '.6f'), 
            "Real time":format(stop - start,'.6f'),}
    else:

        print('NO SOLUTIONS')
        result_dict = {
            "Type": "OR-Tools CP",
            "Data": file_path.split("/")[-1],
            "Number of items": n_items,
            "Bins": '-',  
            "Solver time": format(solver.UserTime(), '.6f'), 
            "Real time": format(stop - start,'.6f'),}
    write_to_xlsx(result_dict)
    

if __name__ == "__main__":
    try:
        # Get input file path
        file_path = sys.argv[1]
    except IndexError:
        # Default input file if file path is not specified
        file_path = 'inputs/0015.txt'

    try:
        # Get input file path
        time_limit = int(sys.argv[2])
    except IndexError:
        # Default input file if file path is not specified
        time_limit = 600
    for i in range(10, 11):
        file_path = f'CLASS\class_01_1.txt'
        print("Reading file: ", file_path.split("/")[-1])
        BPP_CP(file_path, time_limit)
    
        
    
