import sys
import os
import time
import math
import threading
import json
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from zipfile import BadZipFile
from docplex.cp.model import CpoModel
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Global variables
W, H = 0, 0
instance_name = ""
time_out = 600  # seconds (default)
result_dict = {}
optimal_bins = 0
optimal_pos = []
optimal_rot = []
is_timeout = False
start = time.time()

# Unified input reader

def read_file(file_path):
    global instance_name
    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def input_data(file_path):
    global W, H
    input_lines = read_file(file_path)
    n = int(input_lines[0])
    W, H = map(int, input_lines[1].split())
    rectangles = [list(map(int, line.split())) for line in input_lines[2:2+n]]
    return n, rectangles, W, H

def timeout_handler():
    global result_dict, is_timeout
    is_timeout = True
    print(f"\nTimeout reached after {time_out} seconds. Saving current best solution.")
    # Save result as JSON
    result_dict['Status'] = 'TIMEOUT'
    result_dict['Runtime'] = format(time.time() - start, '.3f')
    save_json(result_dict)
    write_to_xlsx(result_dict)
    os._exit(0)

def save_json(result):
    output_path = 'results_' + instance_name + '.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_path}")

def write_to_xlsx(result_dict):
    df = pd.DataFrame([result_dict])
    output_path = 'bpp_cplex_cp.xlsx'
    if os.path.exists(output_path):
        old_df = pd.read_excel(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_excel(output_path, index=False)
    print(f"Result added to Excel file: {os.path.abspath(output_path)}\n")

def display_solution_each_bin(W, H, rectangles, positions, rotations):
    # Group rectangles by bin
    bins = {}
    for i, pos in enumerate(positions):
        bin_id = pos[2] if len(pos) > 2 else 0
        if bin_id not in bins:
            bins[bin_id] = []
        bins[bin_id].append((i, pos, rotations[i]))
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
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def main_solver(file_path, time_limit):
    global result_dict, optimal_bins, optimal_pos, optimal_rot
    n, rectangles, W, H = input_data(file_path)
    k = n
    print(f'Number of items: {n}')
    print(f'Size of bin: {W} {H}')
    model = CpoModel()
    bin_vars = [model.integer_var(0, k-1, f"bin_{i}") for i in range(n)]
    x_vars = [model.integer_var(0, W, f"x_{i}") for i in range(n)]
    y_vars = [model.integer_var(0, H, f"y_{i}") for i in range(n)]
    rot_vars = [model.binary_var(f"rot_{i}") for i in range(n)]
    bin_used = [model.binary_var(f"bin_used_{j}") for j in range(k)]
    widths = []
    heights = []
    for i in range(n):
        w = rectangles[i][0]
        h = rectangles[i][1]
        width_i = model.conditional(rot_vars[i] == 1, h, w)
        height_i = model.conditional(rot_vars[i] == 1, w, h)
        widths.append(width_i)
        heights.append(height_i)
    for i in range(n):
        model.add(x_vars[i] + widths[i] <= W)
        model.add(y_vars[i] + heights[i] <= H)
    for i in range(n-1):
        for j in range(i+1, n):
            same_bin = (bin_vars[i] == bin_vars[j])
            no_overlap = model.logical_or([
                x_vars[i] + widths[i] <= x_vars[j],
                x_vars[j] + widths[j] <= x_vars[i],
                y_vars[i] + heights[i] <= y_vars[j],
                y_vars[j] + heights[j] <= y_vars[i],
            ])
            model.add(model.logical_or([model.logical_not(same_bin), no_overlap]))
    for j in range(k):
        model.add(bin_used[j] == model.logical_or([bin_vars[i] == j for i in range(n)]))
    model.minimize(model.sum(bin_used))
    model.set_parameters(TimeLimit=time_limit)
    print('--------------Solving--------------')
    solve_start = time.time()
    solution = model.solve(TimeLimit=time_limit)
    solve_time = time.time() - solve_start
    result_dict = {
        "Type": "CPLEX CP",
        'Instance': instance_name,
        "Number of items": n,
        "Width": W,
        "Height": H,
        "Optimal_Bins": 0,
        "Runtime": format(solve_time, '.3f'),
        "Status": "UNKNOWN",
    }
    if solution:
        bin_assignments = [int(solution.get_value(bin_vars[i])) for i in range(n)]
        x_positions = [int(solution.get_value(x_vars[i])) for i in range(n)]
        y_positions = [int(solution.get_value(y_vars[i])) for i in range(n)]
        rotations = [int(solution.get_value(rot_vars[i])) for i in range(n)]
        used_bins = set(bin_assignments)
        num_bins_used = len(used_bins)
        positions = []
        for i in range(n):
            positions.append([x_positions[i], y_positions[i], bin_assignments[i]])
        optimal_bins = num_bins_used
        optimal_pos[:] = positions
        optimal_rot[:] = rotations
        result_dict["Optimal_Bins"] = num_bins_used
        result_dict["Status"] = "SUCCESS" if solution.is_solution_optimal() else "FEASIBLE"
        print(f'Number of bins used: {num_bins_used}')
        print(f'Optimal positions: {positions}')
        print(f'Optimal rotations: {rotations}')
        return n, num_bins_used, solve_time, positions, rotations
    else:
        if solution is None:
            result_dict["Status"] = "TIMEOUT"
            print('NO SOLUTION FOUND - TIMEOUT')
        else:
            result_dict["Status"] = "UNSAT"
            print('INFEASIBLE')
        return n, '-', solve_time, [], []

if __name__ == '__main__':
    try:
        # Get time limit
        if len(sys.argv) > 1:
            time_out = int(sys.argv[1])
        else:
            time_out = 100
        file_path = 'inputs/class/CL_1_20_1.txt'  # Change as needed
        print("Reading file:", os.path.basename(file_path))
        timer = threading.Timer(time_out, timeout_handler)
        timer.daemon = True
        timer.start()
        start_main = time.time()
        n, n_bins, solver_time, positions, rotations = main_solver(file_path, time_out)
        result_dict["Real time"] = format(time.time() - start_main, '.3f')
        write_to_xlsx(result_dict)
        save_json(result_dict)
        if positions and rotations:
            display_solution_each_bin(W, H, [[int(x), int(y)] for x, y in read_file(file_path)[2:2+n]], positions, rotations)
        timer.cancel()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        try:
            timer.cancel()
        except:
            pass
        raise
