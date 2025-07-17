#!/usr/bin/env python3
"""
Test the fixed version of OR-TOOLS_MIP_C1.py to verify bug is fixed
"""

import subprocess
import json
import time

def test_instance(instance_name):
    """Test a single instance and check if it violates lower bound"""
    print(f"\n=== Testing {instance_name} ===")
    
    # Run the fixed version with the problematic instance
    cmd = ["python3", "OR-TOOLS_MIP_C1.py", "12"]  # Instance 12 = CL_1_20_1
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("Execution successful!")
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            # Try to find result file
            try:
                with open('results_12.json', 'r') as f:
                    data = json.load(f)
                    print(f"Result data: {data}")
                    
                    if 'N_Bins' in data:
                        bins_found = data['N_Bins']
                        print(f"Bins found: {bins_found}")
                        
                        # Calculate lower bound for CL_1_20_1
                        # We know from previous tests that lower bound = 7
                        lower_bound = 7
                        
                        if bins_found >= lower_bound:
                            print(f"✅ SUCCESS: Bins ({bins_found}) >= Lower bound ({lower_bound})")
                            return True
                        else:
                            print(f"❌ STILL BUGGY: Bins ({bins_found}) < Lower bound ({lower_bound})")
                            return False
                    
            except FileNotFoundError:
                print("No result file found")
                return False
                
        else:
            print(f"Execution failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    # Test the problematic instance
    success = test_instance("CL_1_20_1")
    
    if success:
        print("\n🎉 Bug appears to be FIXED!")
    else:
        print("\n❌ Bug still exists - need more investigation")
