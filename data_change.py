def read_file(n_class, n_items, n_instance):
    n_class = format(n_class, '02d')
    n_items = format(n_items, '03d')
    n_instance = format(n_instance, '02d')
    filepath = f"CLASS/cl_{n_class}_{n_items}_{n_instance}.ins2D"
    print(f"Reading file: {filepath}")
    f = open(filepath, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines

def write(lines, n_class, n_items, n_instance):
    filepath =  f"inputs/CL_{n_class}_{n_items}_{n_instance}.txt"
    with open(filepath, 'w') as f:
        f.write(lines[0] + "\n" + lines[1] + "\n")
        for line in lines[2:]:
            s = line.split()
            f.write(s[1] + " " + s[2] + "\n")
for i in range(1, 11):
    for num in range(20, 101, 20):
        for j in range(1, 11):
            lines = read_file(i, num, j)
            write(lines, i, num, j)




