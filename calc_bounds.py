import os
import sys
import math
def read_file_instance(instance_name):
    possible_paths = [
        f"inputs/BENG/{instance_name}.txt",
        f"inputs/CLASS/{instance_name}.txt",
        f"inputs/{instance_name}.txt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read().splitlines()
    raise FileNotFoundError(f"Cannot find input file for instance {instance_name}")

def calculate_lower_bound(rectangles, W, H):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = W * H
    area_lower_bound = math.ceil(total_area / bin_area)
    
    # Check for items that are too large (no rotation allowed)
    for w, h in rectangles:
        if w > W or h > H:
            return float('inf')  # Infeasible
    
    return max(1, area_lower_bound)

def first_fit_upper_bound(rectangles, W, H):
    bins = []
    def fits(bin_rects, w, h, W, H):
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    for rect in rectangles:
        placed = False
        w, h = rect[0], rect[1]
        if w > W or h > H:
            return float('inf')
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break
        if not placed:
            bins.append([(0, 0, w, h)])
    return len(bins)

def first_fit_upper_bound_r(rectangles, W, H):
    """Finite First-Fit (FFF) upper bound for 2D bin packing with rotation (Berkey & Wang)."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        # For simplicity, try to place at (0, y) for all y up to H-h
        # and check for overlap with all placed rectangles
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
        placed = False
        for bin_rects in bins:
            # Try both orientations in this bin
            for (rw, rh) in [(rect[0], rect[1]), (rect[1], rect[0])]:
                pos = fits(bin_rects, rw, rh, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], rw, rh))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Start a new bin, place at (0,0) in best orientation
            if rect[0] <= W and rect[1] <= H:
                bins.append([(0, 0, rect[0], rect[1])])
            elif rect[1] <= W and rect[0] <= H:
                bins.append([(0, 0, rect[1], rect[0])])
            else:
                # Infeasible rectangle
                return float('inf')
    return len(bins)

def main():
    all_instances_file = "all_instances.txt"
    import pandas as pd
    output_file = "instance_bounds.xlsx"
    if not os.path.exists(all_instances_file):
        print(f"{all_instances_file} not found.")
        sys.exit(1)
    with open(all_instances_file, 'r') as f:
        instances = [line.strip() for line in f if line.strip()]
    rows = []
    for instance_name in instances:
        try:
            lines = read_file_instance(instance_name)
            n_items = int(lines[0])
            W, H = map(int, lines[1].split())
            rectangles = [list(map(int, line.split())) for line in lines[2:2 + n_items]]
            lower = calculate_lower_bound(rectangles, W, H)
            upper = first_fit_upper_bound(rectangles, W, H)
            upper_r = first_fit_upper_bound_r(rectangles, W, H)
            rows.append({
                'instance': instance_name,
                'n': n_items,
                'w': W,
                'h': H,
                'lower_bound': lower,
                'upper_bound': upper, 
                'upper_bound_r': upper_r
            })
            print(f"{instance_name}: lower_bound={lower}, upper_bound={upper}, upper_bound_r={upper_r}")
        except Exception as e:
            rows.append({
                'instance': instance_name,
                'n': None,
                'w': None,
                'h': None,
                'lower_bound': None,
                'upper_bound': None,
                'error': str(e)
            })
            print(f"{instance_name}: ERROR: {e}")
    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
