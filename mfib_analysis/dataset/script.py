import os

# Define folder paths
af2_folder = "af2_mfib_multi_dataset"
native_folder = "native_mfib_dataset"

# Get all .pdb files from af2 folder (without extension)
af2_proteins = set()
if os.path.exists(af2_folder):
    for file in os.listdir(af2_folder):
        if file.endswith('.pdb'):
            protein_name = file.replace('.pdb', '')
            af2_proteins.add(protein_name)

# Get all .pdb files from native folder and check if they exist in af2
if os.path.exists(native_folder):
    for file in os.listdir(native_folder):
        if file.endswith('.pdb'):
            protein_name = file.replace('.pdb', '')
            
            # If this protein is not in af2 folder, remove it
            if protein_name.upper() not in af2_proteins:
                file_path = os.path.join(native_folder, file)
                os.remove(file_path)
                print(f"Removed: {file}")
else:
    print(f"Error: {native_folder} folder not found")

# Optional: Print summary
print(f"\nTotal proteins in af2: {len(af2_proteins)}")
print(f"PDB files removed from native folder (not in af2) shown above")