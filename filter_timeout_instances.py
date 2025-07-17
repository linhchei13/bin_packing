#!/usr/bin/env python3
"""
Script to filter instance names with "time out" status from Excel files
and write them to text files named after each configuration.
"""

import pandas as pd
import os
import glob
from pathlib import Path

def filter_timeout_instances():
    """
    Filter instances with timeout status from Excel files and save to text files.
    """
    # Get all xlsx files in the current directory (excluding backup/copy files)
    xlsx_files = glob.glob("*.xlsx")
    xlsx_files = [f for f in xlsx_files if not any(word in f.lower() for word in ['copy', 'backup', '~'])]
    
    print(f"Found {len(xlsx_files)} Excel files to process")
    
    for xlsx_file in xlsx_files:
        print(f"\nProcessing: {xlsx_file}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(xlsx_file, engine='openpyxl')
            
            # Display columns for debugging
            print(f"  Columns: {list(df.columns)}")
            
            # Find potential status and instance name columns
            status_col = None
            instance_col = None
            
            # Look for status column (various possible names)
            for col in df.columns:
                if any(word in col.lower() for word in ['status', 'result', 'outcome']):
                    status_col = col
                    break
            
            # Look for instance name column
            for col in df.columns:
                if any(word in col.lower() for word in ['instance', 'name', 'problem', 'test']):
                    instance_col = col
                    break
            
            # If standard columns not found, try first few columns
            if status_col is None and len(df.columns) > 1:
                # Often status is in the last column
                status_col = df.columns[-1]
                print(f"  Using last column as status: {status_col}")
            
            if instance_col is None and len(df.columns) > 0:
                # Instance name often in first column
                instance_col = df.columns[0]
                print(f"  Using first column as instance: {instance_col}")
            
            if status_col is None or instance_col is None:
                print(f"  Error: Could not identify status or instance columns")
                continue
            
            # Filter rows with timeout status
            timeout_rows = df[df[status_col].astype(str).str.contains('time.*out|timeout|TIME.*OUT|TIMEOUT|ERROR', case=False, na=False)]
            
            if len(timeout_rows) == 0:
                print(f"  No timeout instances found")
                continue
            
            # Get the instance names with timeout
            timeout_instances = timeout_rows[instance_col].tolist()
            print(f"  Found {len(timeout_instances)} timeout instances")
            
            # Create output filename based on Excel filename
            config_name = Path(xlsx_file).stem
            output_file = f"{config_name}_timeout.txt"
            
            # Write to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                # f.write(f"# Timeout and error instances from {xlsx_file}\n")
                # f.write(f"# Total timeout and error instances: {len(timeout_instances)}\n")
                # f.write(f"# Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for instance in timeout_instances:
                    f.write(f"{instance}\n")
            
            print(f"  Saved {len(timeout_instances)} timeout instances to: {output_file}")
            
        except Exception as e:
            print(f"  Error processing {xlsx_file}: {e}")
            continue

def main():
    """Main function"""
    print("Starting timeout instance filtering...")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # filter_timeout_instances()
    
    print("\n" + "=" * 50)
    print("Filtering completed!")
    
    # Show summary of generated files
    txt_files = glob.glob("*_timeout.txt")
    if txt_files:
        print(f"\nGenerated {len(txt_files)} timeout instance files:")
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                # Count actual instance names (skip comment lines)
                instance_count = sum(1 for line in lines if line.strip() and not line.startswith('#'))
            print(f"  {txt_file} - {instance_count} instances")

if __name__ == "__main__":
    main()
