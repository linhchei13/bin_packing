from pysat.formula import CNF
from pysat.solvers import Glucose3
import sys
import math
from itertools import chain, combinations
from threading import Timer

class TimeoutException(Exception): pass
# if len(sys.argv) < 2:
#     print("Usage: python bin_packing.py inputFile [outputFile] [timeLimit]")
#     sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

file_name = "instances/t120_01.txt"

# Read instance data
file_it = iter(read_integers(file_name))

nb_items = int(next(file_it))
bin_capacity = int(next(file_it))

weights_data = [int(next(file_it)) for i in range(nb_items)]
nb_min_bins = int(math.ceil(sum(weights_data) / float(bin_capacity)))
nb_max_bins = min(nb_items, 2 * nb_min_bins)
max_bins = nb_max_bins
# Declare the optimization model
cnf = CNF()

# Id Set decisions: bin[k] represents the items in bin k
bins = [[i + 1 + k * nb_items for i in range(nb_items)] for k in range(nb_max_bins)]
# Each item must be in one bin and one bin only
for i in range(nb_items):
    cnf.append([bins[k][i] for k in range(nb_max_bins)])# At least one bin
    for k1 in range(nb_max_bins):
        for k2 in range(k1+1, nb_max_bins):
            cnf.append([-bins[k1][i], -bins[k2][i]])  # At most one bin

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# Weight constraint for each bin
for k in range(nb_max_bins):
    for subset in powerset(range(nb_items)):
        if sum(weights_data[i] for i in subset) > bin_capacity:
            # if the sum of the weights of the items in the subset is greater than the bin capacity, then the subset is not v
            cnf.append([-bins[k][i] for i in subset]) 

# Bin k is used if at least one item is in it
bins_used = [i + 1 + nb_items*nb_max_bins for i in range(nb_max_bins)]
for k in range(nb_max_bins):
    for i in range(nb_items):
        cnf.append([-bins[k][i], bins_used[k]])  # If item i is in bin k, bin k is used
    cnf.append([bins[k][i] for i in range(nb_items)] + [-bins_used[k]])  # If bin k is used, there is at least one item in it


# Minimize the number of used bins
timeout = 5 # timeout in seconds

def interrupt(solver):
    solver.interrupt()

with Glucose3(bootstrap_with=cnf.clauses, use_timer=True) as s:
    timer = Timer(timeout, interrupt, [s])
    timer.start()
    try:
        for _ in range(max_bins):
            result = s.solve(assumptions=[-bins_used[k] for k in range(max_bins)])
            print(result)
            if result:
                print("SAT")
                timer.cancel()
                break
            else:
                max_bins -= 1
        else:
            print("UNSAT")
        timer.cancel()
    except TimeoutException:
        print("Timeout")

    # Print solution to console
    if result:
        print("Number of bins:", max_bins)
        model = s.get_model()
        # print(model)
        if model is not None:
            print("Bin assignment:")
            for k in range(nb_max_bins):
                print("Bin", k, ":", [weights_data[i] for i in range(nb_items) if model[bins[k][i] - 1] > 0])
            print("Bin usage:", [k for k in range(nb_max_bins) if model[bins_used[k] - 1] > 0])

