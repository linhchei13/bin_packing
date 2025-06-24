import timeit
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
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
import numpy as np

# Global variables
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
time_out = 600  
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
    global instance_name
    instance_name = file_path.split("/")[-1].split(".")[0]  # Lấy tên file không có phần mở rộng
    with open(file_path) as f:
        data = f.readlines()
        n_packs = int(data[0])
        n_bins = n_packs
        W, H = map(int, data[1].split())
        packs = []
        for i in range(2, n_packs+2):
            line = data[i].split()
            packs.append([int(line[0]), int(line[1])])

    return n_packs, n_bins, packs, W, H

def write_to_xlsx(result_dict):
    df = pd.DataFrame([result_dict])
    output_path = 'bpp_cp.xlsx'
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

def BPP_CP(file_path, time_limit):
    n_items, n_bins, items, W, H = read_input(file_path)
    max_pack_width = max(x[0] for x in items)
    max_pack_height = max(x[1] for x in items)
    print(f"Max pack width: {max_pack_width}")
    start = timeit.default_timer()
    # Creates the model
    model = cp_model.CpModel()
    # Variables
    B = {}
    R = []
    for i in range(n_items):
        # R[i] = 1 iff item i is rotated
        R.append(model.NewBoolVar(f'package_{i}_rotated'))
        for j in range(n_bins):
            # X[i, j] = 1 iff item i is packed in bin j.
            B[i, j] = model.NewBoolVar(f'pack_{i}_in_bin_{j}')

    # Z[j] = 1 iff bin j has been used.
    Z = [model.NewBoolVar(f'bin_{j}_is_used)') for j in range(n_bins)]

    # Width and height of each pack
    width = []
    height = []
    # coordinate 
    x = []
    y = [] 
    for i in range(n_items):
        wi, hi = items[i][0], items[i][1]
        width.append(model.NewIntVar(0, max_pack_width, f'width_{i}'))
        height.append(model.NewIntVar(0, max_pack_height, f'height_{i}'))

        x.append(model.NewIntVar(0, W, f'x_{i}'))
        y.append(model.NewIntVar(0, H, f'y_{i}'))

        # if pack rotated -> switch the height and width
        model.Add(width[i] == items[i][0]).OnlyEnforceIf(R[i].Not())
        model.Add(width[i] == items[i][1]).OnlyEnforceIf(R[i])
        model.Add(height[i] == items[i][1]).OnlyEnforceIf(R[i].Not())
        model.Add(height[i] == items[i][0]).OnlyEnforceIf(R[i])
        
        if wi > W or hi > H:
            model.Add(R[i] == 1)
        # If it is a square (or cannot be rotated because rotated dims are out-of-bound), prevent rotation.
        if wi == hi or (wi > H or hi > W):
            model.Add(R[i] == 0)
    # 
    # Constraint
    # 
    # Each pack can only be placed in one bin
    for i in range(n_items):
        model.Add(sum(B[i, j] for j in range(n_bins)) == 1)
        
    # domain
    for i in range(n_items):
        for j in range(n_bins):
            model.Add(x[i] + width[i]<= W).OnlyEnforceIf(B[i, j])
            model.Add(y[i] + height[i]<= H).OnlyEnforceIf(B[i, j])
            model.Add(x[i] >= 0).OnlyEnforceIf(B[i, j])
            model.Add(y[i] >= 0).OnlyEnforceIf(B[i, j])            
    # If 2 pack in the same bin they cannot overlap
    for i in range(n_items-1):
        for k in range(i+1, n_items):
            a1 = model.NewBoolVar(f"left_{i}_{k}")        
            model.Add(x[i] + width[i] <= x[k]).OnlyEnforceIf(a1)
            model.Add(x[i] + width[i] > x[k]).OnlyEnforceIf(a1.Not())
            a2 = model.NewBoolVar(f'below_{i}_{k}')        
            model.Add(y[i] + height[i] <= y[k]).OnlyEnforceIf(a2)
            model.Add(y[i] + height[i]> y[k]).OnlyEnforceIf(a2.Not())
            a3 = model.NewBoolVar(f'left_{k}_{i}')        
            model.Add(x[k] + width[k] <= x[i] ).OnlyEnforceIf(a3)
            model.Add(x[k] + width[k] > x[i]).OnlyEnforceIf(a3.Not())
            a4 = model.NewBoolVar(f'below_{k}_{i}')        
            model.Add(y[k] + height[k] <= y[i] ).OnlyEnforceIf(a4)
            model.Add(y[k] + height[k] > y[i]).OnlyEnforceIf(a4.Not())
            for j in range(n_bins):
                model.AddBoolOr(a1, a2, a3, a4).OnlyEnforceIf(B[i, j], B[k, j])
    
    # Symmetry breaking constraints.
    # [1] If two rectangles are identical then force a lexicographic order.
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            if items[i] == items[j]:
                model.Add(x[i] <= x[j])
                model.Add(y[i] <= y[j])

    # [2] For rectangles with the maximum width, restrict their horizontal domain.

    for i, rect in enumerate(items):
        wi, hi = rect
        if wi == max_pack_width:
            max_domain = (W - wi) // 2
            model.Add(x[i] <= max_domain)
    # Find which bin has been used               
    for j in range(n_bins):
        b1 = model.NewBoolVar('b')
        model.Add(sum(B[i, j] for i in range(n_items)) == 0).OnlyEnforceIf(b1)
        model.Add(Z[j] == 0).OnlyEnforceIf(b1)
        model.Add(sum(B[i, j] for i in range(n_items)) != 0).OnlyEnforceIf(b1.Not())
        model.Add(Z[j] == 1).OnlyEnforceIf(b1.Not())

    # Objective function
    used = sum(Z[j] for j in range(n_bins))
    model.Minimize(used)
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
        for i in range(n_items):
            for j in range(n_bins):
                if solver.Value(B[i, j]) == 1:
                    optimal_pos.append([solver.Value(x[i]), solver.Value(y[i]), j])
            optimal_rot.append(solver.Value(R[i]))

        optimal_bins = sum(solver.Value(Z[j]) for j in range(n_bins))
        print(f'Number of bin used  : {optimal_bins}')
        # print('----------------Statistics----------------')
        print(f'Status              : {solver.StatusName(status)}')
        print(f'Time limit          : {time_limit}')
        print(f'Running time        : {solver.UserTime()}')
        print(f'Explored branches   : {solver.NumBranches()}')
    
        result_dict = {
            'Instance': instance_name,
            'Runtime': timeit.default_timer() - start,
            'Optimal_Bins': optimal_bins,
            'Status': "SUCCESS",
            'Optimal': "YES" if status == cp_model.OPTIMAL else "NO",
        }
        # display_solution_each_bin(W, H, items, best_pos, best_rot)
    else:
        print('NO SOLUTIONS')
        result_dict = {
            'Instance': instance_name,
            'Runtime': timeit.default_timer() - start,
            'Optimal_Bins': "-",
            'Status': "NO SOLUTIONS",
            'Optimal': "NO",
        }
    write_to_xlsx(result_dict)
    

if __name__ == "__main__":
    try:
        for i in range(1, 11):
            file_path = f'inputs/class/cl_1_40_{i}.txt'
            print("Reading file: ", file_path.split("/")[-1])
            BPP_CP(file_path, time_out)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


