import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Dữ liệu từ bảng
data = {
    "Problem": ["beng01.txt", "beng02.txt", "beng03.txt", "beng04.txt", "beng05.txt",
    "beng06.txt", "beng07.txt", "beng08.txt", "beng09.txt", "beng10.txt",
    "cl_020_01.txt", "cl_020_02.txt", "cl_020_03.txt", "cl_020_04.txt", "cl_020_05.txt",
    "cl_040_01.txt", "cl_040_02.txt", "cl_040_03.txt", "cl_040_04.txt", "cl_040_05.txt",
    "cl_060_01.txt", "cl_060_02.txt", "cl_060_03.txt", "cl_060_04.txt", "cl_060_05.txt",
    "cl_080_01.txt", "cl_080_02.txt", "cl_080_03.txt", "cl_080_04.txt", "cl_080_05.txt",
    "cl_100_01.txt", "cl_100_02.txt", "cl_100_03.txt", "cl_100_04.txt", "cl_100_05.txt"],

    "OR-Tools CP": [1.01, 43.793, 304.288, 308.122, 313.236, 301.427, 309.421, 317.841, 345.293, 600,
    1.559, 1.418, 1.728, 1.31, 2.612, 301.392, 301.565, 18.588, 301.806, 304.586,
    303.049, 306.167, 305.923, 307.635, 304.235, 306.34, 311.92, 310.565, 311.193,
    309.273, 310.326, 313.949, 318.868, 312.237, 313.522
],
    "OR-Tools MIP": [300.46, 600, 600, 600, 600, 600, 600, 600, 600, 600,
    301.279, 301.621, 301.156, 300.653, 300.992, 600, 600, 600, 600, 600,
    600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
    600, 600, 600, 600, 600],
    "$SAT_{optimal}$": [1.479, 4.913, 18.888, 42.948, 98.644, 1.599, 17.753, 89.363, 329.14, 600,
    0.293, 0.167, 0.359, 0.231, 0.228, 2.77, 63.302, 63.387, 63.427, 63.702,
    80.361, 26.219, 148.131, 205.627, 13.205, 191.929, 193.602, 466.394, 217.681, 262.037,
    365.982, 394.067, 352.888, 486.011, 422.4],
    "$SAT_{direct}$": [1.016, 4.83, 16.33, 345.381, 267.341, 2.31, 17.364, 65.615, 170.456, 600,
    1.301, 0.905, 1.216, 0.866, 0.922, 10.051, 105.332, 106.104, 105.756, 106.445,
    152.993, 64.976, 233.48, 344.906, 59.793, 564.813, 590.932, 519.365, 459.775, 600,
    600, 600, 600, 600, 600],
}
# Chuyển thành DataFrame
df = pd.DataFrame(data, index=data["Problem"])

# Vẽ biểu đồ cột
ax = df.plot(kind='bar', figsize=(8, 6), width=0.6)
plt.xlabel("Instances")
plt.ylabel("Time (s)")
plt.title("Performance comparison of different methods on large instances (n >= 100) with time limit 600 seconds")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='center left')
handles, labels = ax.get_legend_handles_labels()
time_limit_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Timeout')
handles.append(time_limit_patch)
plt.legend(handles=handles, loc="upper left")

# Đánh dấu các giá trị 600 bằng đường kẻ ngang
# Đánh dấu các giá trị 600 bằng đường gạch ngang trong cột tương ứng
for bar_group, col_name in zip(ax.containers, df.columns):
    for bar in bar_group:
        if bar.get_height() == 600:
            bar.set_hatch('//')  # Thêm dấu gạch vào cột có giá trị 600


plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
