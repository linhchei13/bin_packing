#!/usr/bin/env python3
"""
Script to move all Excel files with "copy" in their names to a "copy" folder.
"""

import os
import shutil
import glob

def move_copy_files():
    """Move all Excel files with 'copy' in their names to a 'copy' folder"""
    
    # Create copy directory if it doesn't exist
    copy_dir = "copy"
    if not os.path.exists(copy_dir):
        os.makedirs(copy_dir)
        print(f"Created directory: {copy_dir}")
    
    # Find all Excel files with "copy" in their names
    xlsx_files = glob.glob("*.xlsx")
    copy_files = [f for f in xlsx_files if "copy" in f.lower()]
    
    if not copy_files:
        print("No Excel files with 'copy' in their names found.")
        return
    
    print(f"Found {len(copy_files)} Excel files with 'copy' in their names:")
    print("=" * 60)
    
    moved_count = 0
    error_count = 0
    
    for file in copy_files:
        try:
            # Source and destination paths
            src_path = file
            dst_path = os.path.join(copy_dir, file)
            
            # Check if file already exists in destination
            if os.path.exists(dst_path):
                print(f"⚠️  {file} already exists in {copy_dir}/ - skipping")
                continue
            
            # Move file
            shutil.move(src_path, dst_path)
            
            # Get file size for display
            file_size = os.path.getsize(dst_path)
            file_size_kb = file_size / 1024
            
            print(f"✓ Moved: {file} -> {copy_dir}/ ({file_size_kb:.1f} KB)")
            moved_count += 1
            
        except Exception as e:
            print(f"✗ Error moving {file}: {e}")
            error_count += 1
    
    print("=" * 60)
    print(f"Operation completed!")
    print(f"Successfully moved: {moved_count} files")
    print(f"Errors: {error_count} files")
    print(f"Files moved to: {copy_dir}/")
    
    # List files in copy directory
    if moved_count > 0:
        print(f"\nFiles in {copy_dir}/ directory:")
        copy_files_in_dir = os.listdir(copy_dir)
        for file in sorted(copy_files_in_dir):
            if file.endswith('.xlsx'):
                print(f"  - {file}")

def restore_copy_files():
    """Restore Excel files from copy folder back to main directory"""
    copy_dir = "copy"
    
    if not os.path.exists(copy_dir):
        print(f"Directory {copy_dir}/ does not exist.")
        return
    
    # Find all Excel files in copy directory
    copy_files = glob.glob(os.path.join(copy_dir, "*.xlsx"))
    
    if not copy_files:
        print(f"No Excel files found in {copy_dir}/ directory.")
        return
    
    print(f"Found {len(copy_files)} Excel files in {copy_dir}/ directory:")
    print("=" * 60)
    
    restored_count = 0
    error_count = 0
    
    for file_path in copy_files:
        try:
            # Get filename without directory
            filename = os.path.basename(file_path)
            dst_path = filename
            
            # Check if file already exists in main directory
            if os.path.exists(dst_path):
                print(f"⚠️  {filename} already exists in main directory - skipping")
                continue
            
            # Move file back
            shutil.move(file_path, dst_path)
            
            # Get file size for display
            file_size = os.path.getsize(dst_path)
            file_size_kb = file_size / 1024
            
            print(f"✓ Restored: {filename} ({file_size_kb:.1f} KB)")
            restored_count += 1
            
        except Exception as e:
            print(f"✗ Error restoring {os.path.basename(file_path)}: {e}")
            error_count += 1
    
    print("=" * 60)
    print(f"Restoration completed!")
    print(f"Successfully restored: {restored_count} files")
    print(f"Errors: {error_count} files")

def list_copy_files():
    """List all files in copy directory"""
    copy_dir = "copy"
    
    if not os.path.exists(copy_dir):
        print(f"Directory {copy_dir}/ does not exist.")
        return
    
    files = os.listdir(copy_dir)
    xlsx_files = [f for f in files if f.endswith('.xlsx')]
    
    if not xlsx_files:
        print(f"No Excel files found in {copy_dir}/ directory.")
        return
    
    print(f"Excel files in {copy_dir}/ directory ({len(xlsx_files)} files):")
    print("=" * 60)
    
    for file in sorted(xlsx_files):
        file_path = os.path.join(copy_dir, file)
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size / 1024
        print(f"  - {file} ({file_size_kb:.1f} KB)")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) == 1:
        # Default: move copy files
        move_copy_files()
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()
        if command == "move":
            move_copy_files()
        elif command == "restore":
            restore_copy_files()
        elif command == "list":
            list_copy_files()
        else:
            print("Usage:")
            print(f"  {sys.argv[0]}           - Move copy files (default)")
            print(f"  {sys.argv[0]} move      - Move copy files to copy/ folder")
            print(f"  {sys.argv[0]} restore   - Restore files from copy/ folder")
            print(f"  {sys.argv[0]} list      - List files in copy/ folder")
    else:
        print("Usage:")
        print(f"  {sys.argv[0]}           - Move copy files (default)")
        print(f"  {sys.argv[0]} move      - Move copy files to copy/ folder")
        print(f"  {sys.argv[0]} restore   - Restore files from copy/ folder")
        print(f"  {sys.argv[0]} list      - List files in copy/ folder")

if __name__ == "__main__":
    main()
