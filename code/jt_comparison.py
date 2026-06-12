# compare jt distributions in standard and WTA frames

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


def main():
    # 1. SET UP THE ARGUMENT PARSER AND OPEN THE FILES
    # This acts as the bridge allowing Slurm to feed file paths into Python
    parser = argparse.ArgumentParser(description="Compare jt in standard and WTA jet frames")
    parser.add_argument("--input", required=True, type=str, help="Path to the directory of all batches to analyse")
    parser.add_argument("--outdir", required=True, type=str, help="Directory where output should be saved")
    parser.add_argument("--batchnum", required=True, type=int, help="Batch number to analyse")
    args = parser.parse_args()

    file_list = sorted(glob.glob(os.path.join(args.input, f"batch{args.batchnum}", "*.root"))) # Get a list of all root files in that batch directory
    print(f"Found {len(file_list)} files to process in {os.path.join(args.input, f"batch{args.batchnum}")}")
    
    # 2. INITIALISING THINGS
    np.random.seed(42) # set reproducable seed for background
    ROOT.gROOT.SetBatch(True) #Force ROOT into headless batch mode - i.e. turn off pop up graphics
    ROOT.gDirectory.Clear()

    #fastjet stuff
    jet_radius = 1000 #max radius accepted by fastjet is 1000. radius value in prl paper is 0.8 in lab frame.
    wta_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius, fastjet.WTA_pt_scheme) #winner take all definition
    std_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius) #standard E scheme definition

    wta_histograms = {}
    std_histograms = {}

    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]
    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
    
        wta_histograms[bin_key] = ROOT.TH1D(f"WTA_jt_{mult_bin[0]}_{mult_bin[1]}", f"WTA jt distribution for {bin_key};j_{{t}}^{{WTA}};Number of particles", 60, 0, 3)
        std_histograms[bin_key] = ROOT.TH1D(f"STD_jt_{mult_bin[0]}_{mult_bin[1]}", f"Standard jt distribution for {bin_key};j_{t}^{{STD}};Number of particles", 60, 0, 3)

    


    # -------------------------------------------------------------------------
    # DATA STREAMING LOOP OVER MULTIPLE FILES
    # -------------------------------------------------------------------------
    search_pattern = os.path.join(args.input, f"batch{args.batchnum}", "*.root")
    all_files = sorted(glob.glob(search_pattern))
    
    print(f"Found {len(all_files)} total files across batch {args.batchnum} in {args.input}")
    if len(all_files) == 0:
        raise FileNotFoundError(f"No root files found matching pattern: {search_pattern}")

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
        if len(data) == 0:
            continue

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

        wta_constituents = wta_constituents[:, -1] # only need constituents of the relevant recombined jets
        std_constituents = std_constituents[:, -1] 

        # garbage collecting:
        del particle_mask, particles
        del wta_clustered, std_clustered
        gc.collect()
        # ---
    
        # SAVING Nch INFORMATION ----
        # 1. sorting by multiplicity bins
        # we get 1) Nch per jet, 2) number of jets in each bin
        # counting Nch for each jet
        wta_Nch = ak.num(wta_constituents) # number of particles in the recombined jets (before making jet frame cuts, after making lab frame cuts)
        std_Nch = ak.num(std_constituents)

        bin_pass_dict = {}  # dict of true/false masks for whether jets fall in each bin
        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
            
            wta_bin_mask = (wta_Nch >= mult_bin[0]) & (wta_Nch < mult_bin[1])
            std_bin_mask = (std_Nch >= mult_bin[0]) & (std_Nch < mult_bin[1])
            
            bin_pass_dict[bin_key] = {"wta" : wta_bin_mask , "std" : std_bin_mask}

        # garbage collecting
        del wta_bin_mask, std_bin_mask, bin_key
        gc.collect()
        # -----

        # TRANSFORMING TO JET FRAME ----
        ## 1. eta_star
        wta_thetas = wta_jets.deltaangle(wta_constituents)  # angle theta that constituents make with the jet axis
        std_thetas = std_jets.deltaangle(std_constituents)  # range of 0 to pi, => eta range +infty to 0

        wta_theta_mask = wta_thetas > 0.01347569            # i.e. cutting out eta star > 5
        std_theta_mask = std_thetas > 0.01347569

        wta_thetas = wta_thetas[wta_theta_mask]
        std_thetas = std_thetas[std_theta_mask]

        wta_eta_star = -np.log( np.tan(wta_thetas / 2))     # eta star formula for acceptable thetas
        std_eta_star = -np.log( np.tan(std_thetas / 2))

        ## 2. jt
        wta_jt = wta_constituents.p[wta_theta_mask] * np.sin(wta_thetas) # jt = p sin theta_star, only for valid thetas
        std_jt = std_constituents.p[std_theta_mask] * np.sin(std_thetas)


        # sorting the jet frame particles by jet multiplicity
        mult_sorted_jets = {}
        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
            mult_sorted_jets[bin_key] = {
                "wta": wta_jt[bin_pass_dict[bin_key]["wta"]], 
                "std": std_jt[bin_pass_dict[bin_key]["std"]]
            }

        # garbage collecting
        del wta_eta_star, std_eta_star, wta_thetas, std_thetas, wta_theta_mask, std_theta_mask
        gc.collect()

        # ----

        #Fill histograms for this chunk:
        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"

            # Filter the jet-frame particles down to just the jets in this bin
            binned_wta_parts = mult_sorted_jets[bin_key]["wta"]
            binned_std_parts = mult_sorted_jets[bin_key]["std"]
            
            # --- Fill jt hists ---
            wta_jt = ak.to_numpy(ak.flatten(binned_wta_parts, axis=None))
            if len(wta_jt) > 0:
                wta_histograms[bin_key].FillN(len(wta_jt), wta_jt.astype(np.float64),  np.ones_like(wta_jt, dtype=np.float64))

            std_jt = ak.to_numpy(ak.flatten(binned_std_parts, axis=None))
            if len(std_jt) > 0:
                std_histograms[bin_key].FillN(len(std_jt), std_jt.astype(np.float64),  np.ones_like(std_jt, dtype=np.float64))

            del binned_wta_parts, binned_std_parts
            gc.collect()

        del mult_sorted_jets, bin_pass_dict
        gc.collect()

    print("All streaming chunks processed. Persistent histograms populated completely.")


    # SAVING THE OUTPUT TO A SINGLE COMBINED COHESIVE ROOT FILE
    os.makedirs(args.outdir, exist_ok=True)
    full_output_path = os.path.join(args.outdir, f"jt_comparison_batch{args.batchnum}.root")
    
    print(f"Opening output file to save all results: {full_output_path}")
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
        bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

        # Write out items
        wta_histograms[bin_key].Write()
        std_histograms[bin_key].Write()

        print(f"Written batch {args.batchnum} histograms for multiplicity bin: {bin_name}")

    out_file.Close()
    print(f"Results for batch {args.batchnum} written and saved")

    print("Job successfully completed")
    # END ----


# running:
if __name__ == "__main__":
    main()
