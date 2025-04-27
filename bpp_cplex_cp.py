from docplex.cp.model import CpoModel
import sys
import pandas as pd
import os
import time
from openpyxl import load_workbook
from openpyxl import Workbook
from zipfile import BadZipFile
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime 
from matplotlib import pyplot as plt
W, H = 0, 0
global result_dict
result_dict = {}

def input_data(file_path):
    global W, H
    data = {}
    f = open(file_path,'r')
    
    data['size_item'] = []
    data['size_bin'] = []
    n = int(f.readline())
    W, H = map(int, f.readline().split())
    data['size_bin'].append([W, H])
    for i in range(1,n+1):
        line = f.readline().split()
        data['size_item'].append([int(line[0]),int(line[1])])
    
    return n,data,W,H


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

def main_solver(file_path, time_limit):
    n,data,W,H = input_data(file_path)
    k = n
    print('Number of items:',n)
    print('Size of bin:',W,H)
    
    # Create CP model
    model = CpoModel()
    
    # Define variables
    # For each item, create a tuple (bin, x, y, rot)
    # bin: which bin the item is in (0 to k-1)
    # x, y: coordinates of the bottom-left corner
    # rot: 1 if rotated by 90 degrees, 0 otherwise
    bin_vars = [model.integer_var(0, k-1, f"bin_{i}") for i in range(n)]
    x_vars = [model.integer_var(0, W, f"x_{i}") for i in range(n)]
    y_vars = [model.integer_var(0, H, f"y_{i}") for i in range(n)]
    rot_vars = [model.binary_var(f"rot_{i}") for i in range(n)]
    
    # Create variables for bin usage
    bin_used = [model.binary_var(f"bin_used_{j}") for j in range(k)]
    
    # Compute width and height based on rotation
    widths = []
    heights = []
    for i in range(n):
        w = data['size_item'][i][0]
        h = data['size_item'][i][1]
        # Width is w if not rotated, h if rotated
        width_i = model.conditional(rot_vars[i] == 1, h, w)
        # Height is h if not rotated, w if rotated
        height_i = model.conditional(rot_vars[i] == 1, w, h)
        widths.append(width_i)
        heights.append(height_i)
    
    
    
    # Ensure items stay within bin boundaries
    for i in range(n):
        model.add(x_vars[i] + widths[i] <= W)
        model.add(y_vars[i] + heights[i] <= H)
    
    # Non-overlap constraints between items in the same bin
    for i in range(n-1):
        for j in range(i+1, n):
            # If items i and j are in the same bin
            same_bin = (bin_vars[i] == bin_vars[j])
            
            # Then they must not overlap
            # Either i is to the left of j, or i is to the right of j,
            # or i is below j, or i is above j
            no_overlap = model.logical_or([
                x_vars[i] + widths[i] <= x_vars[j],  # i is to the left of j
                x_vars[j] + widths[j] <= x_vars[i],  # j is to the left of i
                y_vars[i] + heights[i] <= y_vars[j],  # i is below j
                y_vars[j] + heights[j] <= y_vars[i],  # j is below i
            ])
            
            # If same_bin is true, then no_overlap must be true
            model.add(model.logical_or([model.logical_not(same_bin), no_overlap]))
    
    # Connect bin_used variables to bin_vars
    for j in range(k):
        # bin_used[j] is true if any item is in bin j
        model.add(bin_used[j] == model.logical_or([bin_vars[i] == j for i in range(n)]))
    
    
    # Objective: minimize number of bins used
    model.minimize(model.sum(bin_used))
    
    # Set time limit
    model.set_parameters(TimeLimit=time_limit)
    
    # Solve model
    print('--------------Solving--------------')
    start = time.time()
    solution = model.solve(TimeLimit=time_limit)
    stop = time.time()
    solve_time = stop - start
    print("Solving time: ", solve_time)

    
    global result_dict
    result_dict = {
        "Type": "CPLEX CP",
        'Problem': file_path.split("/")[-1],
        "Number of items": n,
        "Width": W,
        "Height": H,
        "Number of bins": 0,
        "Time": format(solve_time, '.6f'),
        "Result": "UNKNOWN",
    }
    
    if solution:
        print('--------------Solution Found--------------')
        # Extract solution
        bin_assignments = [solution.get_value(bin_vars[i]) for i in range(n)]
        x_positions = [solution.get_value(x_vars[i]) for i in range(n)]
        y_positions = [solution.get_value(y_vars[i]) for i in range(n)]
        rotations = [solution.get_value(rot_vars[i]) for i in range(n)]
        
        print(rotations)
        # Count used bins
        used_bins = set(bin_assignments)
        num_bins_used = len(used_bins)
        
        # Print solution
        for j in sorted(used_bins):
            print(f"Bin {j+1}:")
            items_in_bin = [i for i in range(n) if bin_assignments[i] == j]
            for i in items_in_bin:
                w = data['size_item'][i][0]
                h = data['size_item'][i][1]
                print(f'  Put item {i+1} {data["size_item"][i]} with rotation {rotations[i]} at ({x_positions[i]},{y_positions[i]})')
            print("-------")
            # Uncomment to display solution
            # display_solution((W, H), 
            #                 [data['size_item'][i] for i in items_in_bin],
            #                 [(x_positions[i], y_positions[i]) for i in items_in_bin], 
            #                 [rotations[i] for i in items_in_bin])
        
        result_dict["Number of bins"] = num_bins_used
        result_dict["Result"] = "SAT"
        print(f'Number of bins used: {num_bins_used}')
        print('----------------Statistics----------------')
        
        if solution.is_solution_optimal():
            print('Status: OPTIMAL')
        else:
            print('Status: FEASIBLE')
            
        print(f'Time limit: {time_limit}')
        print(f'Running time: {solve_time}')               
        return n, num_bins_used, format(solve_time, '.6f')
    else:
        if solution is None:
            result_dict["Result"] = "UNKNOWN"
            print('NO SOLUTION FOUND - TIMEOUT')
            return n, '-', format(solve_time, '.6f')
        else:
            result_dict["Result"] = "UNSAT"
            print('INFEASIBLE')
            return n, '-', format(solve_time, '.6f')


if __name__ == '__main__':

        try:
            # Get time limit
            time_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        except IndexError:
            time_limit = 100
           
        # Create solver
        file_path = f'input_data/test.txt'
        print("Reading file: ", file_path.split("/")[-1])
        start = time.time()
        n, n_bins, solver_time = main_solver(file_path, time_limit)
        stop = time.time()
        
        result_dict["Real time"] = format(stop - start, '.6f')
        write_to_xlsx(result_dict)
