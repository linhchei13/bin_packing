import subprocess

# Danh sách tên session (cũng là tên file .py sẽ chạy)
session_names = [
                    "bpp_ms_r_sb", "bpp_ms_s_c1", "bpp_ms_c1", "bpp_ms_s_r_sb",
                  # "cplex_mip_c1", "cplex_mip_r_sb",
                  # "gurobi_mip_c1", "gurobi_mip_r_sb",
                  "or-tools_mip_c1", "or-tools_mip_r_sb",
                  "orr-tools_cp_c1", "or-tools_cp_r_sb",
                ]

# Lặp qua từng session
for name in session_names:
    
    py_file = f"{name.upper()}.py"
    
    # Câu lệnh tmux để tạo session và chạy file Python
    cmd = f'tmux new-session -d -s {name} "python3 {py_file}"'

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Created tmux session '{name}' running '{py_file}'")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create session '{name}': {e}")
