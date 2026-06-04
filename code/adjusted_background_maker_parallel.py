# makes background for given mult bin for the given cluster type
# adjusted the split handling

import argparse
import glob
import ROOT
import os
import gc
import uproot
import numpy as np

def generate_and_fill_bkg_chunked(hist, probabilities, flat_eta, flat_phi, d_eta, d_phi, n_target_pairs, chunk_size=10000000):
    """
    Generates and fills background pairs in manageable chunks to keep memory usage low 
    and prevent PyROOT alpha binding buffer overhead.
    """
    PI = np.pi
    pairs_processed = 0
    
    while pairs_processed < n_target_pairs:
        # Determine the size of the current batch
        this_chunk = min(chunk_size, n_target_pairs - pairs_processed)
        
        # 1. Vectorized chunk sampling
        idx1 = np.random.choice(len(probabilities), size=this_chunk, p=probabilities)
        idx2 = np.random.choice(len(probabilities), size=this_chunk, p=probabilities)
        
        # 2. Continuous un-binning with localized uniform smudge
        eta1 = flat_eta[idx1] + np.random.uniform(-d_eta/2, d_eta/2, size=this_chunk)
        phi1 = flat_phi[idx1] + np.random.uniform(-d_phi/2, d_phi/2, size=this_chunk)
        eta2 = flat_eta[idx2] + np.random.uniform(-d_eta/2, d_eta/2, size=this_chunk)
        phi2 = flat_phi[idx2] + np.random.uniform(-d_phi/2, d_phi/2, size=this_chunk)
        
        # 3. Two-particle relative kinematics
        deta = np.abs(eta1 - eta2)
        dphi = np.arccos(np.cos(phi1 - phi2))
        
        # 4. Explicitly cast to float64 and create weights for PyROOT FillN
        deta_f64 = deta.astype(np.float64)
        dphi_f64 = dphi.astype(np.float64)
        weights = np.ones(this_chunk, dtype=np.float64)
        
        # 5. Direct 6-Way Fill into the ROOT Histogram Canvas
        hist.FillN(this_chunk, deta_f64, dphi_f64, weights)
        hist.FillN(this_chunk, -deta_f64, dphi_f64, weights)
        hist.FillN(this_chunk, deta_f64, -dphi_f64, weights)
        hist.FillN(this_chunk, -deta_f64, -dphi_f64, weights)
        hist.FillN(this_chunk, deta_f64, -dphi_f64 + 2 * PI, weights)
        hist.FillN(this_chunk, -deta_f64, -dphi_f64 + 2 * PI, weights)
        
        pairs_processed += this_chunk

        print(f"processed {pairs_processed} out of {n_target_pairs} pairs")
        
        # Immediate inner-loop memory cleanup
        del idx1, idx2, eta1, phi1, eta2, phi2, deta, dphi, deta_f64, dphi_f64, weights
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Run 2-Particle Correlation Background Generator")
    parser.add_argument("--input", required=True, type=str, help="Path to input merged file")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save output file")
    parser.add_argument("--clustertype", required=True, type=str, help="WTA or STD")
    parser.add_argument("--multbin", required=True, type=str, help="Multiplicity bin (in form 0_25)")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of parallel splits for this mult bin and cluster type")
    parser.add_argument("--split_id", type=int, default=0, help="Sub-job split identifier, goes from 0 to num_splits - 1")
    #parser.add_argument("--split_fraction", type=float, default=1.0, help="Fraction of pairs to generate (e.g., 0.10 for 10%)")
    args = parser.parse_args()

    ROOT.gDirectory.Clear()
    ROOT.gROOT.SetBatch(True)

    merged_filepath = args.input
    merged_file_root = ROOT.TFile.Open(merged_filepath, "READ")
    merged_file_up = uproot.open(merged_filepath)

    prefix = args.clustertype
    bin_name = args.multbin

    print(f"Cluster type: {prefix}")
    
    # choosing file name depending on whether split or not
    if args.num_splits > 1:
        split_filename = f"bkg_{prefix}_{bin_name}_split{args.split_id}.root"
        full_output_path = os.path.join(args.outdir, split_filename)
    else:
        full_output_path = os.path.join(args.outdir, f"{prefix}_merged_backgrounds_{bin_name}.root")

    # create output path
    out_file = ROOT.TFile(full_output_path, "RECREATE")
    print(f"Output file to save results: {full_output_path}")
    
    print(f"\n--- Processing Multiplicity Bin: {bin_name} ---")

    # Load raw signal objects to read entry statistics
    hSignal = merged_file_root.Get(f"{prefix}_sig_{bin_name}")

    # Extract true pair entries (accounting for your signal script's 6-way duplication)
    NENT = int(hSignal.GetEntries() // 6)

    # total number of bg pairs to generate across splits
    total_bg_pairs = 10 * NENT

    # 2. Distribute pairs cleanly, allocating the remainder to the first few split_ids
    base_pairs = total_bg_pairs // args.num_splits
    remainder = total_bg_pairs % args.num_splits

    n_bg_pairs = base_pairs + (1 if args.split_id < remainder else 0) #1 extra pair per job to make up the remainder


    # Initialize background canvas templates
    hBckrnd = hSignal.Clone(f"{prefix}_bkg_{bin_name}")
    hBckrnd.Reset()

    # ==========================================
    # STEP 1: PROCESSING THE GIVEN FRAME
    # ==========================================
    epd = merged_file_up[f"{prefix}_epd_{bin_name}"]
    counts, eta_edges, phi_edges = epd.to_numpy()

    eta_centers = (eta_edges[:-1] + eta_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    d_eta = eta_edges[1] - eta_edges[0]
    d_phi = phi_edges[1] - phi_edges[0]

    eta_grid, phi_grid = np.meshgrid(eta_centers, phi_centers, indexing='ij')
    flat_eta = eta_grid.ravel()
    flat_phi = phi_grid.ravel()
    probabilities = counts.ravel().astype(np.float64)

    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities /= prob_sum
        print(f" -> Generating {n_bg_pairs} {prefix} background pairs...")
        generate_and_fill_bkg_chunked(
            hBckrnd, probabilities, flat_eta, flat_phi, 
            d_eta, d_phi, n_bg_pairs
        )

    # Clear WTA structural arrays out of memory scope
    del counts, eta_grid, phi_grid, flat_eta, flat_phi, probabilities
    gc.collect()

    # Save finished background structures to file
    out_file.cd()
    hBckrnd.Write()
    print(f"Successfully wrote backgrounds for bin {bin_name}")

    # Final loop cleanup
    del hBckrnd, hSignal
    gc.collect()

    out_file.Close()
    merged_file_root.Close()
    print("Job fully completed.")

if __name__ == "__main__":
    main()