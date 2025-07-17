import os
import pandas as pd
import logging
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_excel_by_instance(excel_file, instance_name, updates):
    """Update specific instance in Excel file with new data - remove duplicates, keep first"""
    try:
        if not os.path.exists(excel_file):
            logger.error(f"Excel file {excel_file} does not exist")
            return False
            
        # Load existing Excel file
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        if 'Instance' not in df.columns:
            logger.error("No 'Instance' column found in Excel file")
            return False
            
        # Find the instance
        mask = df['Instance'] == instance_name
        if not mask.any():
            logger.warning(f"Instance {instance_name} not found in Excel file")
            return False
            
        # Get the first occurrence index
        first_occurrence = df.index[mask].tolist()[0]
        
        # Update only the first occurrence
        for column, value in updates.items():
            if column not in df.columns:
                df[column] = ''  # Add new column if it doesn't exist
            df.loc[first_occurrence, column] = value
            
        # Remove duplicate occurrences (keep only first)
        df = df.drop_duplicates(subset=['Instance'], keep='first')
        
        # Save back to Excel
        df.to_excel(excel_file, index=False, engine='openpyxl')
        logger.info(f"Updated first occurrence of instance {instance_name} and removed duplicates")
        return True
        
    except Exception as e:
        logger.error(f"Error updating Excel file: {e}")
        return False

def get_instance_from_excel(excel_file, instance_name):
    """Get data for specific instance from Excel file"""
    try:
        if not os.path.exists(excel_file):
            return None
            
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        if 'Instance' not in df.columns:
            return None
            
        mask = df['Instance'] == instance_name
        if mask.any():
            return df[mask].iloc[0].to_dict()
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return None

def add_or_update_instance(excel_file, instance_data):
    """Add new instance or update existing one in Excel file - remove duplicates, keep first"""
    try:
        instance_name = instance_data.get('Instance')
        if not instance_name:
            logger.error("No instance name provided")
            return False
            
        # Check if file exists
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file, engine='openpyxl')
            
            # Check if instance exists
            if 'Instance' in df.columns and instance_name in df['Instance'].values:
                # Find all occurrences of this instance
                mask = df['Instance'] == instance_name
                
                # Get the first occurrence index
                first_occurrence = df.index[mask].tolist()[0]
                
                # Update only the first occurrence
                for column, value in instance_data.items():
                    if column not in df.columns:
                        df[column] = ''
                    df.loc[first_occurrence, column] = value
                
                # Remove duplicate occurrences (keep only first)
                df = df.drop_duplicates(subset=['Instance'], keep='first')
                
                logger.info(f"Updated first occurrence of instance {instance_name} and removed duplicates")
            else:
                # Add new instance
                df_new = pd.DataFrame([instance_data])
                df = pd.concat([df, df_new], ignore_index=True)
                logger.info(f"Added new instance {instance_name}")
        else:
            # Create new file
            df = pd.DataFrame([instance_data])
            logger.info(f"Created new Excel file with instance {instance_name}")
            
        # Save to Excel
        df.to_excel(excel_file, index=False, engine='openpyxl')
        return True
        
    except Exception as e:
        logger.error(f"Error updating Excel file: {e}")
        return False

def remove_instance(excel_file, instance_name):
    """Remove specific instance from Excel file"""
    try:
        if not os.path.exists(excel_file):
            logger.error(f"Excel file {excel_file} does not exist")
            return False
            
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        if 'Instance' not in df.columns:
            logger.error("No 'Instance' column found in Excel file")
            return False
            
        # Remove all occurrences of the instance
        df = df[df['Instance'] != instance_name]
        
        # Save back to Excel
        df.to_excel(excel_file, index=False, engine='openpyxl')
        logger.info(f"Removed all occurrences of instance {instance_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error removing instance from Excel file: {e}")
        return False
results = {
    'Instance': 'CL_3_80_10',
    'Runtime': 897.8680106909987
    
}
remove_instance('BPP_MS_R_SB.xlsx', 'CL_3_60_9')