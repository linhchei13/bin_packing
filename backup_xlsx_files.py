#!/usr/bin/env python3
"""
Script to create backups for all Excel files in the workspace.
Creates backup copies with timestamp suffix.
"""

import os
import shutil
import glob
from datetime import datetime
import sys

def create_xlsx_backups():
    """Create backups for all Excel files in the current directory"""
    
    # Get current timestamp for backup naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all Excel files in current directory
    xlsx_files = glob.glob("*.xlsx")
    
    if not xlsx_files:
        print("No Excel files found in the current directory.")
        return
    
    # Create backup directory
    backup_dir = f"backups_xlsx_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"Creating backups in directory: {backup_dir}")
    print("=" * 50)
    
    success_count = 0
    error_count = 0
    
    for xlsx_file in xlsx_files:
        try:
            # Skip files that are already backups (contain 'copy' or 'backup')
            if any(word in xlsx_file.lower() for word in ['copy', 'backup']):
                print(f"Skipping {xlsx_file} (already a backup file)")
                continue
                
            # Create backup filename
            base_name = os.path.splitext(xlsx_file)[0]
            backup_filename = f"{base_name}_backup.xlsx"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file to backup location
            shutil.copy2(xlsx_file, backup_path)
            
            # Get file size for display
            file_size = os.path.getsize(xlsx_file)
            file_size_kb = file_size / 1024
            
            print(f"✓ Backed up: {xlsx_file} -> {backup_filename} ({file_size_kb:.1f} KB)")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error backing up {xlsx_file}: {e}")
            error_count += 1
    
    print("=" * 50)
    print(f"Backup completed!")
    print(f"Successfully backed up: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Backup directory: {backup_dir}")
    
    # Create a summary file
    summary_file = os.path.join(backup_dir, "backup_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Excel Files Backup Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {len(xlsx_files)}\n")
        f.write(f"Successfully backed up: {success_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"\nBacked up files:\n")
        
        for xlsx_file in xlsx_files:
            if not any(word in xlsx_file.lower() for word in ['copy', 'backup']):
                f.write(f"- {xlsx_file}\n")
    
    print(f"Summary saved to: {summary_file}")

def restore_from_backup(backup_dir=None):
    """Restore Excel files from a backup directory"""
    if backup_dir is None:
        # Find the most recent backup directory
        backup_dirs = glob.glob("backups_xlsx_*")
        if not backup_dirs:
            print("No backup directories found.")
            return
        backup_dir = max(backup_dirs)
    
    if not os.path.exists(backup_dir):
        print(f"Backup directory '{backup_dir}' not found.")
        return
    
    print(f"Restoring from backup directory: {backup_dir}")
    
    backup_files = glob.glob(os.path.join(backup_dir, "*.xlsx"))
    
    if not backup_files:
        print("No Excel files found in backup directory.")
        return
    
    success_count = 0
    
    for backup_file in backup_files:
        try:
            # Extract original filename
            filename = os.path.basename(backup_file)
            if '_backup_' in filename:
                original_name = filename.split('_backup_')[0] + '.xlsx'
            else:
                original_name = filename
            
            # Copy back to current directory
            shutil.copy2(backup_file, original_name)
            print(f"✓ Restored: {original_name}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error restoring {backup_file}: {e}")
    
    print(f"Restoration completed! Restored {success_count} files.")

def list_backups():
    """List all available backup directories"""
    backup_dirs = glob.glob("backups_xlsx_*")
    
    if not backup_dirs:
        print("No backup directories found.")
        return
    
    print("Available backup directories:")
    for backup_dir in sorted(backup_dirs):
        timestamp = backup_dir.replace("backups_xlsx_", "")
        # Parse timestamp for display
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Count files in backup
            xlsx_count = len(glob.glob(os.path.join(backup_dir, "*.xlsx")))
            
            print(f"  {backup_dir} - {formatted_date} ({xlsx_count} files)")
        except:
            print(f"  {backup_dir}")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        # Default: create backup
        create_xlsx_backups()
    elif len(sys.argv) == 2:
        command = sys.argv[1].lower()
        if command == "backup":
            create_xlsx_backups()
        elif command == "restore":
            restore_from_backup()
        elif command == "list":
            list_backups()
        else:
            print("Usage:")
            print(f"  {sys.argv[0]}           - Create backup (default)")
            print(f"  {sys.argv[0]} backup    - Create backup")
            print(f"  {sys.argv[0]} restore   - Restore from latest backup")
            print(f"  {sys.argv[0]} list      - List available backups")
    elif len(sys.argv) == 3:
        command = sys.argv[1].lower()
        if command == "restore":
            backup_dir = sys.argv[2]
            restore_from_backup(backup_dir)
        else:
            print("Usage:")
            print(f"  {sys.argv[0]} restore <backup_dir>  - Restore from specific backup")
    else:
        print("Usage:")
        print(f"  {sys.argv[0]}           - Create backup (default)")
        print(f"  {sys.argv[0]} backup    - Create backup")
        print(f"  {sys.argv[0]} restore   - Restore from latest backup")
        print(f"  {sys.argv[0]} restore <backup_dir>  - Restore from specific backup")
        print(f"  {sys.argv[0]} list      - List available backups")

if __name__ == "__main__":
    main()
