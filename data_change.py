def read_file(n_instance):
    filepath = "CLASS/CLASS/cl_01_100_0{}.ins2D".format(n_instance)
    f = open(filepath, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines

def write(lines, n):
    filepath = "input_data/class/CL_100_0{}.txt".format(n)
    with open(filepath, 'w') as f:
        f.write(lines[0] + "\n" + lines[1] + "\n")
        for line in lines[2:]:
            s = line.split()
            f.write(s[1] + " " + s[2] + "\n")
for i in range(1, 6):
    write(read_file(i), i)




