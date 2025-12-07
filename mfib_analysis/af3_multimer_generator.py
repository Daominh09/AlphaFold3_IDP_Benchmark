
import csv
import json
import requests
import time
from typing import List, Dict, Tuple


def fetch_sequences(pdb_id: str) -> Tuple[str, str]:
    seq1 = ""
    seq2 = ""
    
    try:
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            
            sequences = []
            current_seq = []
            
            for line in lines:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            if current_seq:
                sequences.append(''.join(current_seq))
            
            if len(sequences) >= 2:
                seq1 = sequences[0]
                seq2 = sequences[1]
            elif len(sequences) == 1:
                seq1 = sequences[0]
                seq2 = ""
         
            
            print(f"    Found {len(sequences)} sequence(s)")
    
    except Exception as e:
        print(f"    Error: {e}")
    
    return seq1, seq2


def parse_chains(chain_str: str) -> int:
    """
    Determine if homodimer (A;A-2) or heterodimer (A;B)
    
    Returns:
        1 for homodimer, 2 for heterodimer
    """
    chains = chain_str.split(';')
    if len(chains) == 2:
        base1 = chains[0].strip().replace('-2', '').replace('-3', '')
        base2 = chains[1].strip().replace('-2', '').replace('-3', '')
        if base1 == base2:
            return 1  # Homodimer
        if base2=='':
            return 1  # Homodimer (single chain repeated)
        
    elif len(chains) == 1:
        return 1  # Homodimer (single chain repeated)
    
    return 2  # Heterodimer


def create_jobs(csv_path: str) -> List[Dict]:
    """Create AF3 jobs from CSV"""
    jobs = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader, 1):
            pdb_id = row['PDB_ID'].strip()
            chain_str = row['Chains'].strip()
            organism = row.get('Organism', '').strip()
            
            print(f"[{idx}/90] {pdb_id.upper()}")
            
            seq1, seq2 = fetch_sequences(pdb_id)
            # if pdb_id.upper() == '1T6F':
            #     print(seq1, seq2)
                
            # if not seq1:
            #     print(f"    ✗ Failed")
            #     continue
            
            chain_type = parse_chains(chain_str)
            
            # if pdb_id.upper() == '1T6F':
            #     break
            # Simple job name
            job = {
                "name": f"{pdb_id.upper()}",
                "modelSeeds": [],
                "sequences": [],
                "dialect": "alphafoldserver",
                "version": 1
            }
            
            if chain_type == 1:
                # Homodimer - one sequence, count=2
                job["sequences"].append({
                    "proteinChain": {
                        "sequence": seq1,
                        "count": 2
                    }
                })
                print(f"    ✓ Homodimer: {len(seq1)} residues x 2")
                
            elif chain_type == 2 and not seq2:
                # Homodimer - one sequence, count=2
                job["sequences"].append({
                    "proteinChain": {
                        "sequence": seq1,
                        "count": 2
                    }
                })
                print(f"    ✓ Homodimer: {len(seq1)} residues x 2")
            
            else:
                # Heterodimer - two sequences
                if seq1:
                    job["sequences"].append({
                        "proteinChain": {
                            "sequence": seq1,
                            "count": 1
                        }
                    })
                    print(f"    ✓ Chain 1: {len(seq1)} residues")
                
                if seq2:
                    job["sequences"].append({
                        "proteinChain": {
                            "sequence": seq2,
                            "count": 1
                        }
                    })
                    print(f"    ✓ Chain 2: {len(seq2)} residues")
            
            if job["sequences"]:
                jobs.append(job)
            
            time.sleep(0.5)
    
    return jobs


def main():
    """Main function"""
    csv_path = 'mfib_subset_90.csv'
    
    print("=" * 60)
    print("AF3 JSON Generator")
    print("=" * 60)
    
    jobs = create_jobs(csv_path)
    
    print(f"\n✅ Created {len(jobs)} jobs")
    
 
    jobs_part1 = jobs[:30]
    jobs_part2 = jobs[30:37]
    jobs_part3= jobs[37:]
    
    # Write part 1
    with open('af3_jobs_part1.json', 'w') as f:
        json.dump(jobs_part1, f, indent=2)
    print(f"💾 af3_jobs_part1.json: {len(jobs_part1)} jobs")
    
    # Write part 2
    with open('af3_jobs_part2.json', 'w') as f:
        json.dump(jobs_part2, f, indent=2)
    print(f"💾 af3_jobs_part2.json: {len(jobs_part2)} jobs")
    
    with open('af3_jobs_part3.json', 'w') as f:
        json.dump(jobs_part3, f, indent=2)
    print(f"💾 af3_jobs_part3.json: {len(jobs_part3)} jobs")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()