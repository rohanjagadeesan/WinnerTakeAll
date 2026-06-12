# Compare Nch between the two cluster types


# IMPORTS ---
import argparse
import glob
import ROOT
import math
import fastjet
import os
import gc
import awkward as ak
import uproot
import numpy as np
import vector
vector.register_awkward()
# ---



def main():
    
    # 1. SET UP THE ARGUMENT PARSER AND OPEN THE FILES
    parser = argparse.ArgumentParser(description="Run Nch comparison")
    parser.add_argument("--input", required=True, type=str, help="Path to the directory of all batches to analyse")
    parser.add_argument("--outdir", required=True, type=str, help="Directory where output should be saved")
    parser.add_argument("--batchnum", required=True, type=int, help="Batch number to analyse")
    args = parser.parse_args()


    # copied from signal maker:
    # -------------------------------------------------------------------------
    # DATA STREAMING LOOP OVER MULTIPLE FILES
    # -------------------------------------------------------------------------
    # Discover files globally (handles wildcards or folder inputs)

    all_files = sorted(glob.glob(os.path.join(args.input, f"batch{args.batchnum}", "*.root"), recursive=True))
    
    print(f"Found {len(all_files)} total files across all batches in {args.input}")
    if len(all_files) == 0:
        raise FileNotFoundError(f"No root files found")

    # --- NEW: PRE-SCREENING FILTER FOR INCOMPLETE/CORRUPTED FILES ---
    required_branches = {"genJetPt", "genJetEta", "genDau_pt", "genDau_eta", "genDau_phi", "genDau_chg"}
    valid_files = []
    
    print("Pre-screening files to verify required tree and branches...")
    for filepath in all_files:
        try:
            with uproot.open(filepath) as f:
                if "trackTree" in f:
                    tree = f["trackTree"]
                    # check if all required branches are subsets of the tree keys
                    if required_branches.issubset(tree.keys()):
                        valid_files.append(filepath)
                    else:
                        missing = required_branches - set(tree.keys())
                        print(f"Skipping {os.path.basename(filepath)}: missing branches {missing}")
                else:
                    print(f"Skipping {os.path.basename(filepath)}: 'trackTree' not found")
        except Exception as e:
            # Captures files that are partially written, zero-byte size, or corrupted
            print(f"Skipping {os.path.basename(filepath)}: error opening file ({e})")
            
    print(f"Retained {len(valid_files)} / {len(all_files)} valid files for processing.")
    if len(valid_files) == 0:
        raise FileNotFoundError("None of the discovered files contain a valid 'trackTree' with the necessary branches.")

    # 2. Append the TTree name ONLY to verified discovered file paths
    tree_paths = [f"{filepath}:trackTree" for filepath in valid_files]
    print("Beginning vectorized streaming iteration over valid files...")



    # 2. INITIALISING THINGS
    ROOT.gROOT.SetBatch(True) #Force ROOT into headless batch mode - i.e. turn off pop up graphics

    # Jet definitions
    jet_radius = 1000 #max radius accepted by fastjet is 1000. radius value in prl paper is 0.8 in lab frame.
    wta_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius, fastjet.WTA_pt_scheme) #winner take all definition
    std_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius) #standard E scheme definition

    # analysis bins
    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    # Initialize batch histograms :
    # the difference in raw Nch values:
    blind_Nch_diff_hist = ROOT.TH1D("blind_Nch_diff", "Per-jet difference in N_{ch};N_{ch}^{STD}-N_{ch}^{WTA};Number of jets", 31, -15.5, 15.5)

    # difference in constituents:
    # for particles in standard but not in wta, and vice versa:
    const_unique_to_std = ROOT.TH1D("const_unique_to_std", "Particles in STD but NOT WTA;N_{tracks};Number of Jets", 17, -0.5, 16.5)
    const_unique_to_wta = ROOT.TH1D("const_unique_to_wta", "Particles in WTA but NOT STD;N_{tracks};Number of Jets", 17, -0.5, 16.5)

    # Plain Nch histograms
    wta_Nch_hist = ROOT.TH1D("WTA_Nch", "WTA multiplicity distribution;N_{ch}^{WTA};Number of jets", 130, 0, 130)
    std_Nch_hist = ROOT.TH1D("STD_Nch", "Standard multiplicity distribution;N_{ch}^{STD};Number of jets", 130, 0, 130)
    

    chunk_counter = 0
    # Pass the filtered list of paths directly to uproot.iterate
    for data in uproot.iterate(tree_paths, expressions=["genJetPt", "genJetEta", "genDau_pt", "genDau_eta", "genDau_phi", "genDau_chg"], step_size="150 MB"):
        chunk_counter += 1
        print(f"--- Processing Data Chunk #{chunk_counter} ---")

        # PRELIM ANALYSIS (Cuts) ---
        # lab frame jet cuts
        jetEtaCut = 1.6 # from PRL paper page 2
        jetPtCut = 550  # from PRL paper page 2
        data = data[(data.genJetPt > jetPtCut) & (abs(data.genJetEta) < jetEtaCut)] # applying cuts
        data = data[ak.num(data.genJetPt) > 0] # removing empty jets
        if len(data) == 0: continue

        # flattening to the relevant jets
        # Flatten the event/jet axes so that axis 0 is simply a list of all passing jets
        jet_dau_pt = ak.flatten(data.genDau_pt, axis=1)
        jet_dau_eta = ak.flatten(data.genDau_eta, axis=1)
        jet_dau_phi = ak.flatten(data.genDau_phi, axis=1)
        jet_dau_chg = ak.flatten(data.genDau_chg, axis=1)
        energies = jet_dau_pt * np.cosh(jet_dau_eta) # calculates total momentum, approximately equal to particle energy. p = pt * cosh(eta)

        # Build a jagged array of 4-vectors for all particles inside all jets
        particles = ak.zip({
            "pt": jet_dau_pt,
            "eta": jet_dau_eta,
            "phi": jet_dau_phi,
            "E": energies 
        }, with_name="Momentum4D")

        del data    # don't need jet level info anymore now that cuts have been made
        gc.collect()

        # 2. lab frame particle cuts
        particlePtCut = 0.3 #from prl paper page 2
        particleEtaCut = 2.4 #from prl paper page 2

        # particle selection
        particle_mask = (particles.pt > particlePtCut) & (abs(particles.eta) < particleEtaCut) & (jet_dau_chg != 0) #charged particles, meeting pt and eta cuts
        particles = particles[particle_mask] #now only valid particles left
        if len(particles) == 0 or ak.sum(ak.num(particles)) == 0:
            continue


        # 3. clustering the lab frame particles
        wta_clustered = fastjet.ClusterSequence(particles, wta_def)
        std_clustered = fastjet.ClusterSequence(particles, std_def)

        wta_jets = fastjet.sorted_by_pt(wta_clustered.inclusive_jets()) # the recombined wta jets
        std_jets = fastjet.sorted_by_pt(std_clustered.inclusive_jets()) # the recombined std jets

        wta_constituents = wta_clustered.constituents() #the constituent particles of the recombined jets
        std_constituents = std_clustered.constituents()

        # using leading jets.
        # fastjet sometimes recombines one jet into two or more, so in those cases i need to use the highest pt recombined one. 
        wta_jets = wta_jets[:, -1] #slicing to be only the leading jet in the recombined lists. Other items in the list will only have a couple of soft particles
        std_jets = std_jets[:, -1]
        wta_constituents = wta_constituents[:, -1]  # only need constituents of the relevant recombined jets
        std_constituents = std_constituents[:, -1]

        # ---
    
        # SAVING Nch INFORMATION ----
        # counting Nch for each jet
        wta_Nch = ak.num(wta_constituents) # number of particles in the recombined jets (before making jet frame cuts, after making lab frame cuts)
        std_Nch = ak.num(std_constituents)
        blind_diff = std_Nch - wta_Nch
        

        # Extract Unique Constituents via Cartesian Overlap Masks
        pairs_std = ak.cartesian([particles.phi, std_constituents.phi], axis=1, nested=True)
        p_phi, c_phi = ak.unzip(pairs_std)
        is_in_std = ak.any(abs(p_phi - c_phi) < 1e-5, axis=-1)

        pairs_wta = ak.cartesian([particles.phi, wta_constituents.phi], axis=1, nested=True)
        p_phi_wta, c_phi_wta = ak.unzip(pairs_wta)
        is_in_wta = ak.any(abs(p_phi_wta - c_phi_wta) < 1e-5, axis=-1)

        n_std_not_wta = ak.sum(is_in_std & ~is_in_wta, axis=1)
        n_wta_not_std = ak.sum(is_in_wta & ~is_in_std, axis=1)

        # Convert arrays to numpy double types for ROOT FillN
        wta_Nch_np = ak.to_numpy(wta_Nch).astype(np.float64)
        std_Nch_np = ak.to_numpy(std_Nch).astype(np.float64)
        blind_diff_np = ak.to_numpy(blind_diff).astype(np.float64)
        n_std_not_wta_np = ak.to_numpy(n_std_not_wta).astype(np.float64)
        n_wta_not_std_np = ak.to_numpy(n_wta_not_std).astype(np.float64)
        
        weights = np.ones(len(wta_Nch_np), dtype=np.float64)

        # Fill Histograms
        wta_Nch_hist.FillN(len(wta_Nch_np), wta_Nch_np, weights)
        std_Nch_hist.FillN(len(std_Nch_np), std_Nch_np, weights)
        blind_Nch_diff_hist.FillN(len(blind_diff_np), blind_diff_np, weights)
        const_unique_to_std.FillN(len(n_std_not_wta_np), n_std_not_wta_np, weights)
        const_unique_to_wta.FillN(len(n_wta_not_std_np), n_wta_not_std_np, weights)

        del wta_clustered, std_clustered, wta_constituents, std_constituents, pairs_std, pairs_wta
        gc.collect()

        
        # -----
    
    # Post-Loop Processing
    print(f"File loop completed - executing post processing")

    # Plot 5: Subtraction (STD - WTA)
    diff_distribution_hist = std_Nch_hist.Clone("STD_minus_WTA_distribution")
    diff_distribution_hist.SetTitle("Multiplicity Distribution Difference (Standard - WTA);N_{ch};#Delta Jets")
    diff_distribution_hist.Add(wta_Nch_hist, -1.0)

    # Plot 6: Ratio (STD / WTA)
    ratio_distribution_hist = std_Nch_hist.Clone("STD_over_WTA_distribution")
    ratio_distribution_hist.SetTitle("Multiplicity Distribution Ratio (Standard / WTA);N_{ch};Ratio")
    ratio_distribution_hist.Divide(wta_Nch_hist)

    # Write out everything to one file
    os.makedirs(args.outdir, exist_ok=True)
    full_output_path = os.path.join(args.outdir, f"Cluster_type_comparison_batch{args.batchnum}.root")
    
    print(f"Opening output file to save all results: {full_output_path}")
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    blind_Nch_diff_hist.Write()
    const_unique_to_std.Write()
    const_unique_to_wta.Write()
    wta_Nch_hist.Write()
    std_Nch_hist.Write()
    diff_distribution_hist.Write()
    ratio_distribution_hist.Write()
    out_file.Close()

    print(f"Analysis successfully saved to {full_output_path}")


# running
if __name__ == "__main__":
    main()
        

