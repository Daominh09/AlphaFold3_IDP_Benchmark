#!/usr/bin/env python
"""
PyMOL script to calculate Buried Surface Area (BSA) for protein complexes
Adds BSA scores directly to existing AF2 and AF3 CSV files
"""

import os
import csv
import pandas as pd
from pymol import cmd

def calculate_bsa(structure_file, file_type='pdb'):
    """
    Calculate buried surface area for a protein complex
    
    Args:
        structure_file: Path to PDB or CIF file
        file_type: 'pdb' or 'cif' to indicate file format
        
    Returns:
        float: BSA value or None if error
    """
    # Load the structure
    structure_name = os.path.splitext(os.path.basename(structure_file))[0]
    
    try:
        if file_type == 'cif':
            cmd.load(structure_file, structure_name, format='cif')
        else:
            cmd.load(structure_file, structure_name)
    except Exception as e:
        print(f"  Error loading {structure_file}: {str(e)}")
        return None
    
    # Get SASA for the complex
    sasa_complex = cmd.get_area(structure_name)
    
    # Get all chains in the structure
    chains = cmd.get_chains(structure_name)
    
    if len(chains) == 0:
        print(f"  Warning: No chains found in {structure_name}")
        cmd.delete(structure_name)
        return None
    
    # Calculate SASA for each individual chain
    total_individual_sasa = 0
    
    for chain in chains:
        selection = f"{structure_name} and chain {chain}"
        sasa = cmd.get_area(selection)
        total_individual_sasa += sasa
    
    # Calculate BSA
    # BSA = (Sum of individual chain SASAs - Complex SASA) / 2
    bsa = (total_individual_sasa - sasa_complex) / 2
    
    # Clean up
    cmd.delete(structure_name)
    
    return round(bsa, 2)

def process_csv_with_bsa(csv_file, structure_folder, file_extension, file_type, pdb_id_column='pdb_id'):
    """
    Add BSA column to existing CSV file
    
    Args:
        csv_file: Path to CSV file to update
        structure_folder: Folder containing structure files
        file_extension: '.pdb' or '.cif'
        file_type: 'pdb' or 'cif' for PyMOL
        pdb_id_column: Name of the column containing PDB IDs
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found")
        return
    
    if not os.path.exists(structure_folder):
        print(f"Error: Structure folder {structure_folder} not found")
        return
    
    # Read the CSV file
    print(f"\nReading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if pdb_id_column not in df.columns:
        print(f"Error: Column '{pdb_id_column}' not found in {csv_file}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"Found {len(df)} rows")
    
    # Add BSA column if it doesn't exist
    if 'bsa' not in df.columns:
        df['bsa'] = None
    
    # Calculate BSA for each row
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        pdb_id = row[pdb_id_column]
        
        # Construct the file path
        structure_file = os.path.join(structure_folder, f"{pdb_id}{file_extension}")
        
        if not os.path.exists(structure_file):
            print(f"  [{idx+1}/{len(df)}] Warning: File not found: {pdb_id}{file_extension}")
            error_count += 1
            continue
        
        print(f"  [{idx+1}/{len(df)}] Calculating BSA for: {pdb_id}")
        
        try:
            bsa = calculate_bsa(structure_file, file_type)
            if bsa is not None:
                df.at[idx, 'bsa'] = bsa
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"  Error processing {pdb_id}: {str(e)}")
            error_count += 1
    
    # Save the updated CSV
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Updated {csv_file}")
    print(f"  Successfully calculated: {success_count}")
    print(f"  Errors/Missing: {error_count}")

def main():
    """Main function"""
    # Configuration - MODIFY THESE AS NEEDED
    config = {
        'af2': {
            'csv_file': 'dataset/af2v3_dockq_data.csv',  # Your AF2 CSV file
            'structure_folder': 'dataset/alphafold2_files',  # Folder with .pdb files
            'file_extension': '.pdb',
            'file_type': 'pdb',
            'pdb_id_column': 'pdb_id'  # Column name containing PDB IDs
        },
        'af3': {
            'csv_file': 'dataset/af3_dockq_data.csv',  # Your AF3 CSV file
            'structure_folder': 'dataset/alphafold3_files',  # Folder with .cif files
            'file_extension': '.cif',
            'file_type': 'cif',
            'pdb_id_column': 'pdb_id'  # Column name containing PDB IDs
        }
    }
    
    # Initialize PyMOL in quiet mode
    cmd.reinitialize()
    
    print("="*70)
    print("BSA Calculator - Adding BSA scores to CSV files")
    print("="*70)
    
    # Process AlphaFold2 CSV
    print("\n" + "-"*70)
    print("Processing AlphaFold2 data...")
    print("-"*70)
    process_csv_with_bsa(
        config['af2']['csv_file'],
        config['af2']['structure_folder'],
        config['af2']['file_extension'],
        config['af2']['file_type'],
        config['af2']['pdb_id_column']
    )
    
    # Process AlphaFold3 CSV
    print("\n" + "-"*70)
    print("Processing AlphaFold3 data...")
    print("-"*70)
    process_csv_with_bsa(
        config['af3']['csv_file'],
        config['af3']['structure_folder'],
        config['af3']['file_extension'],
        config['af3']['file_type'],
        config['af3']['pdb_id_column']
    )
    
    print("\n" + "="*70)
    print("Done! BSA column added to both CSV files")
    print("="*70)

if __name__ == "__main__":
    main()