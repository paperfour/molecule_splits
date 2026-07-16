import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from pathlib import Path
import time


# Construct the adjacency matrix for the hiv dataset graph, where the nodes are moleules
# and the edges are the tanimoto similarity between molecules
def generate_adjacency_matrix(data_df, save_dir, mfp_radius=2, mfp_size=2048, verbose=True):

    if verbose:
        print("WARNING: Generating the adjacency matrix will take a very large amout of memory for larger datasets")

    # Preload the morgan fingerprints of all the molecules
    if verbose:
        print(f"Preloading MFPs of radius={mfp_radius}, size={mfp_size}")
    
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=mfp_radius, fpSize=mfp_size)
    mfps = [gen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in data_df["smiles"]]

    
    NUM_MOLS = len(mfps)
    if verbose:
        print(f"Creating empty {NUM_MOLS} x {NUM_MOLS} matrix")
    
    adjacency_mat = np.full((NUM_MOLS, NUM_MOLS), np.nan)
    
    np.fill_diagonal(adjacency_mat, 0)

    
    if verbose:
        print(f"Populating matrix with CSO values")
    
    TOTAL_COMPARISONS = int((NUM_MOLS ** 2 - NUM_MOLS) / 2)
    current = -1
    
    for i, mfp1 in enumerate(mfps):
        for j, mfp2 in enumerate(mfps):
            
            # Skip if comparing two identical molecules or the lower triangular portion
            if i >= j:
                continue

            current += 1

            if verbose and (current % 100000 == 0 or current == TOTAL_COMPARISONS):
                print("\r" + f"{round((current * 100) / TOTAL_COMPARISONS, 2)} %\t\tCurrently on {current} / {TOTAL_COMPARISONS}\t\t", end="")
            
            tan_sim = DataStructs.TanimotoSimilarity(mfp1, mfp2)
            adjacency_mat[i, j] = tan_sim
            adjacency_mat[j, i] = tan_sim
                
            
    if verbose:
        print()
        print(f"Completed! Saving into directory {save_dir}")

    # Save the matrix
    save_path = Path(f"{save_dir}/cso_adjacency_matrix.npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(save_path, adjacency_mat)

    if verbose:
        print("Saved")

    return save_path


# Method for saving a massive amount of tanimioto similarities for visualization purposes. It will random sample to save space
def save_tan_sim(tan_sims, save_name, verbose=False, save_size=100000):
    
    if verbose:
        print("\nSaving...")
        
    target_size = save_size
    if len(tan_sims) > target_size:

        if verbose:
            print("\nAggregating data")
        
        save_arr = np.random.choice(tan_sims, save_size)
        
        np.savetxt(f"{save_name}_similarity_dist.csv", save_arr, delimiter=",", header="similarity", comments="")
        
        if verbose:
            print("\nSaved")
    else:
        np.savetxt(f"{save_name}_similarity_dist.csv", tan_sims, delimiter=",", header="similarity", comments="")


# Method to find the cross split overlap between two groups (idxs)
# TODO: Refactor to simply "adj_mat_cso"
def tanimoto_cso(adj_mat, idxs1, idxs2, verbose = False, save_name = None, save_size = 100000):

    num_comps = len(idxs1) * len(idxs2)
    
    tan_sims = np.full(num_comps, np.nan)

    start = time.time()

    for i, mol1 in enumerate(idxs1):
        for j, mol2 in enumerate(idxs2):
            
            tan_sims[i * len(idxs2) + j] = adj_mat[mol1, mol2]
    
            if verbose and (i * len(idxs2) + j) % 10000000 == 0:
                
                elapsed = time.time() - start
                prog = (i * len(idxs2) + j) / num_comps

                #  To prevent division by zero error
                if prog == 0:
                    prog = 0.0000001
                    
                print(f"\rCompleted {i * len(idxs2) + j} / {num_comps}\t\t{prog * 100:.2f} %\t\tEstimated wait: {elapsed / prog - elapsed:.2f} s          ", end = "")

    if verbose:
        print("\nFinished!")

    if save_name != None:
        save_tan_sim(tan_sims, save_name, verbose=verbose, save_size=save_size)

    return np.mean(tan_sims)

# Will find the average tanimoto similarities of the highest k cross split comparisons
def tanimoto_cso_k_max(idxs1, idxs2, k, verbose = False, save_name = None, save_size = 100000):

    num_comps = len(idxs1) * len(idxs2)
    
    tan_sims = np.full(num_comps, np.nan)

    start = time.time()

    for i, mol1 in enumerate(idxs1):
        for j, mol2 in enumerate(idxs2):

            tan_sim = ADJACENCY_MAT[mol1, mol2]

            assert tan_sim >= 0
            assert tan_sim <= 1
            
            tan_sims[i * len(idxs2) + j] = tan_sim

    
            if verbose and (i * len(idxs2) + j) % 10000000 == 0:
                
                elapsed = time.time() - start
                prog = (i * len(idxs2) + j) / num_comps

                #  To prevent division by zero error
                if prog == 0:
                    prog = 0.0000001
                    
                print(f"\rCompleted {i * len(idxs2) + j} / {num_comps}\t\t{prog * 100:.2f} %\t\tEstimated wait: {elapsed / prog - elapsed:.2f} s          ", end = "")

    if verbose:
        print("\nFinished!")

    if save_name != None:
        save_tan_sim(tan_sims, save_name, verbose=verbose, save_size=save_size)
        
    max_k_indices = np.argpartition(tan_sims, -k)[-k:]
    return np.mean(tan_sims[max_k_indices])


# Converts two molecules to their morgan fingerprints and then computes tanimoto similarity between these two vectors
# Does not use preprocessed adjacency matrix... good for verfication
def tanimoto_similarity(mol1, mol2, radius = 2, fp_size = 2048, gen = None):

    if gen == None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = fp_size)

    mfp1 = mol1
    # String --> mol --> MFP
    if isinstance(mol1, str):
        mfp1 = gen.GetFingerprint(Chem.MolFromSmiles(mol1))
    # mol --> MFP
    elif isinstance(mol1, rdchem.Mol):
        mfp1 = gen.GetFingerprint(mol1) 

    mfp2 = mol2
    # String --> mol --> MFP
    if isinstance(mol2, str):
        mfp2 = gen.GetFingerprint(Chem.MolFromSmiles(mol2))
    # mol --> MFP
    elif isinstance(mol2, rdchem.Mol):
        mfp2 = gen.GetFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(mfp1, mfp2)


# A simple test to somewhat verify an adjacency matrix
def test_adj_mat_random(adj_mat, source_df):
    
    # Ensure the matrix is up to par
    print("Random testing adj mat...")
    
    N = 10000
    
    for i, j in [(random.randint(0, len(source_df) - 1), random.randint(0, len(source_df) - 1)) for _ in range(N)]:
    
        mol1 = i
        mol2 = j
        
        # Adjacency matrix
        adj1 = adj_mat[mol1, mol2]
        adj2 = adj_mat[mol2, mol1]
        
        # Manual method
        man1 = tanimoto_similarity(source_df.at[mol1, "smiles"], source_df.at[mol2, "smiles"])
        man2 = tanimoto_similarity(source_df.at[mol2, "smiles"], source_df.at[mol1, "smiles"])
        
        assert adj1 == adj2
        assert adj1 == man1
        assert man1 == man2
        
    print("Test passed!")
