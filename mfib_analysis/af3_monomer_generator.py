
import csv
import json
import requests
import time
from typing import List, Dict, Tuple


def fetch_all_sequences(pdb_id: str) -> List[Tuple[str, str]]:
    sequences = []
    
    try:
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            
            current_chain = ""
            current_seq = []
            
            for line in lines:
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_seq and current_chain:
                        sequences.append((current_chain, ''.join(current_seq)))

                    header = line[1:]  # Remove '>'
                    parts = header.split('|')
                    if parts:
                        # Extract chain from first part (e.g., "1ABC_A")
                        first_part = parts[0]
                        if '_' in first_part:
                            current_chain = first_part.split('_')[1]
                        else:
                            current_chain = f"chain_{len(sequences) + 1}"
                    
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            # Don't forget the last sequence
            if current_seq and current_chain:
                sequences.append((current_chain, ''.join(current_seq)))
            
            print(f"    Found {len(sequences)} chain(s)")
    
    except Exception as e:
        print(f"    Error: {e}")
    
    return sequences


def create_monomer_jobs(csv_path: str) -> List[Dict]:
    """Create AF3 monomer jobs from CSV - one job per chain"""
    jobs = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)
        
        for idx, row in enumerate(rows, 1):
            pdb_id = row['Complex_ID'].strip()
            
            print(f"[{idx}/{total}] {pdb_id.upper()}")
            
            sequences = fetch_all_sequences(pdb_id)
            
            if not sequences:
                print(f"    ✗ No sequences found")
                continue
            
            # Track unique sequences to avoid duplicates
            seen_sequences = set()
            
            for chain_id, seq in sequences:
                # Skip duplicate sequences (common in homo-oligomers)
                if seq in seen_sequences:
                    print(f"    ⊘ Chain {chain_id}: duplicate sequence, skipping")
                    continue
                
                seen_sequences.add(seq)
                
                # Create a monomer job for this chain
                job = {
                    "name": f"{pdb_id.upper()}_{chain_id}",
                    "modelSeeds": [],
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": seq,
                                "count": 1
                            }
                        }
                    ],
                    "dialect": "alphafoldserver",
                    "version": 1
                }
                
                jobs.append(job)
                print(f"    ✓ Chain {chain_id}: {len(seq)} residues (monomer)")
            
            time.sleep(0.5)
    
    return jobs


def main():
    """Main function"""
    csv_path = 'results/interface_ligand_rmsd_af2_af3_comparison.csv'
    
    print("=" * 60)
    print("AF3 JSON Generator - Monomer Version")
    print("=" * 60)
    
    jobs = create_monomer_jobs(csv_path)
    
    print(f"\n✅ Created {len(jobs)} monomer jobs")
    
    # Split into 3 parts
    n = len(jobs)
    jobs_part1 = jobs[:n//3]
    jobs_part2 = jobs[n//3:n*2//3]
    jobs_part3 = jobs[n*2//3:]
    
    # Write part 1
    with open('af3_monomer_jobs_part1.json', 'w') as f:
        json.dump(jobs_part1, f, indent=2)
    print(f"💾 af3_monomer_jobs_part1.json: {len(jobs_part1)} jobs")
    
    # Write part 2
    with open('af3_monomer_jobs_part2.json', 'w') as f:
        json.dump(jobs_part2, f, indent=2)
    print(f"💾 af3_monomer_jobs_part2.json: {len(jobs_part2)} jobs")
    
    # Write part 3
    with open('af3_monomer_jobs_part3.json', 'w') as f:
        json.dump(jobs_part3, f, indent=2)
    print(f"💾 af3_monomer_jobs_part3.json: {len(jobs_part3)} jobs")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
