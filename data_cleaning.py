import pandas as pd
from pathlib import Path
import json
from rdkit import Chem


CHNOPS = {6, 1, 7, 8, 15, 16}

HALIDES = {9, 17, 35, 53}

def clean_data(data_path, save_dir, mols_to_exclude=[], print_errors=False):

    data_df = pd.read_csv(data_path)
    
    # For marking progress
    max_idx = len(data_df) - 1
    
    
    # Start a json that will store which molecules were edited/removed
    info = {"source": str(data_path), "removed" : [], "invalid" : [], "canonicalized" : [], "exceptional" : []}
    
    # Cannonicalize and filter duplicate/invalid smiles
    for idx, smiles in enumerate(data_df["smiles"]):

        if idx % 100 == 0 or idx == max_idx:
            print("\r" + f"Cleaning index {idx}/{max_idx}", end="")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            cannonical_smiles = Chem.MolToSmiles(mol)
            #     If specified to be removed...            Or there is a duplicate
            if (cannonical_smiles in mols_to_exclude) or (cannonical_smiles in data_df.loc[0:idx - 1, "smiles"]):

                info["removed"].append((idx, smiles))
                data_df.drop(index = idx, inplace = True)

            # Check for any exceptional atoms in the molecule
            atoms = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
            remaining = atoms.difference(CHNOPS)
            remaining = remaining.difference(HALIDES)
            # If there are any exceptional atoms, remove the molecule
            if remaining:
                info["exceptional"].append((idx, smiles))
                data_df.drop(index = idx, inplace = True)
            
                
            # SMILES is not the RDKit cannon
            elif cannonical_smiles != smiles:
                info["canonicalized"].append((idx, smiles))
                data_df.at[idx, "smiles"] = cannonical_smiles
                
        # SMILES is invalid to make a molecule
        except Exception as e:
            if print_errors:
                print()
                print(e)
            info["invalid"].append((idx, smiles))
            data_df.drop(index = idx, inplace = True)

    print()
    print(f"Done! Saving data into folder {save_dir}")

    # Save the filter info
    info_save_path = Path(f"{save_dir}/data_info.json")
    info_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_save_path, "w") as f:
        json.dump(info, f, indent=2)    

    # Save the cleaned data
    data_save_path = Path(f"{save_dir}/clean_data.csv")
    data_df.to_csv(data_save_path, index = False)

    return data_save_path


# Make sure a loaded dataframe is all proper emiles
def check_validity(data_df):
    
    print("Checking validity...")

    duplicates = data_df[data_df.duplicated(keep=False)]
    
    assert len(duplicates) == 0, f"Duplicate found! See df: {duplicates}"
    
    for smiles in data_df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)  
        Chem.MolToSmiles(mol)
    
    print("If nothing was printed, everything is valid!")
