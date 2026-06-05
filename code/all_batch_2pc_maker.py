# this file takes in 
# 1. a directory containing directories for each batch
# 2. an output directory where the results are saved
# 3. the batch number to analyse all files for

# output: A root file containing (for both wta and standard, for all files in the batch) 
# - signal, EPD, total Nch, and number of jets

# skips invalid files


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
    # This acts as the bridge allowing Slurm to feed file paths into Python
    parser = argparse.ArgumentParser(description="Run WTA 2-Particle Correlation on NOTS Cluster")
    parser.add_argument("--input", required=True, type=str, help="Path to the directory of all batches to analyse")
    parser.add_argument("--outdir", required=True, type=str, help="Directory where output should be saved")
    parser.add_argument("--batchnum", required=True, type=int, help="Batch number to analyse")
    args = parser.parse_args()

    file_list = sorted(glob.glob(os.path.join(args.input, f"batch{args.batchnum}", "*.root"))) # Get a list of all root files in that batch directory
    print(f"Found {len(file_list)} files to process in {os.path.join(args.input, f"batch{args.batchnum}")}")
    
    # 2. INITIALISING THINGS
    np.random.seed(42) # set reproducable seed for background
    ROOT.gROOT.SetBatch(True) #Force ROOT into headless batch mode - i.e. turn off pop up graphics

    # Jet definitions
    jet_radius = 0.8 #max radius accepted by fastjet is 1000. radius value in prl paper is 0.8 in lab frame.
    wta_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius, fastjet.WTA_pt_scheme) #winner take all definition
    std_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius) #standard E scheme definition

    # analysis bins
    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    # Initialize persistent histogram storage and tracking variables BEFORE the loop
    wta_histograms = {}
    std_histograms = {}

    # Accumulators for global normalization math across all file streams
    num_jets_dict = {f"{b[0]} < Nch < {b[1]}": {"wta": 0.0, "std": 0.0} for b in analysis_bins} # number of jets per bin
    total_Nch_sum_dict = {f"{b[0]} < Nch < {b[1]}": {"wta": 0.0, "std": 0.0} for b in analysis_bins} # Nch per jet bin

    # Setup histogram bin specs
    PI = np.pi 
    eta_bins, eta_min, eta_max = 41, -6.15, 6.15    
    phi_bins = 33
    phi_min = -(math.pi/2.0) + (math.pi/32.0)
    phi_max = (3*math.pi/2.0) + (math.pi/32.0)

    # Book empty histograms once up front
    ROOT.gDirectory.Clear()
    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
        safe_name = bin_key.replace(" ", "_").replace("<", "lt").replace(">", "gt")
    
        wta_histograms[bin_key] = {
            "signal": ROOT.TH2D(f"WTA_sig_{safe_name}", f"WTA Signal ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max),
            "background": ROOT.TH2D(f"WTA_bg_{safe_name}", f"WTA Background ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max),
            "epd": ROOT.TH2D(f"WTA_epd_{safe_name}", f"WTA EPD ({bin_key});#eta*;#phi*", 150, 0, 10, 120, -4, 4)
        }
        
        std_histograms[bin_key] = {
            "signal": ROOT.TH2D(f"STD_sig_{safe_name}", f"Standard Signal ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max),
            "background": ROOT.TH2D(f"STD_bg_{safe_name}", f"Standard Background ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max),
            "epd": ROOT.TH2D(f"STD_epd_{safe_name}", f"Standard EPD ({bin_key});#eta*;#phi*", 150, 0, 10, 120, -4, 4)
        }

    # -------------------------------------------------------------------------
    # DATA STREAMING LOOP OVER MULTIPLE FILES
    # -------------------------------------------------------------------------
    file_pattern = os.path.join(args.input, "*.root:trackTree")
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
        # we get 1) Nch per jet, 2) avg Nch per bin, 3) number of jets in each bin
        # counting Nch for each jet
        wta_Nch = ak.num(wta_constituents) # number of particles in the recombined jets (before making jet frame cuts, after making lab frame cuts)
        std_Nch = ak.num(std_constituents)

        bin_pass_dict = {}  # dict of true/false masks for whether jets fall in each bin
        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
            
            wta_bin_mask = (wta_Nch >= mult_bin[0]) & (wta_Nch < mult_bin[1])
            std_bin_mask = (std_Nch >= mult_bin[0]) & (std_Nch < mult_bin[1])
            
            bin_pass_dict[bin_key] = {"wta" : wta_bin_mask , "std" : std_bin_mask}
            
            # Incrementally accumulate global jet counts across all streams
            num_jets_dict[bin_key]["wta"] += float(ak.sum(wta_bin_mask)) # summing number of Trues gives number of passing jets
            num_jets_dict[bin_key]["std"] += float(ak.sum(std_bin_mask))
            
            # Accumulate raw Nch sums to compute global averages later
            total_Nch_sum_dict[bin_key]["wta"] += float(ak.sum(wta_Nch[wta_bin_mask]))
            total_Nch_sum_dict[bin_key]["std"] += float(ak.sum(std_Nch[std_bin_mask]))

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

        jt_lower = 0.3 #could also be 0.5
        wta_jt_mask = (wta_jt > jt_lower)  &  (wta_jt < 3)  #0.3<jt<3
        std_jt_mask = (std_jt > jt_lower)  &  (std_jt < 3)

        wta_jt = wta_jt[wta_jt_mask]                # only valid jt's
        wta_eta_star = wta_eta_star[wta_jt_mask]    # only etas corresponding to valid jt's
        std_jt = std_jt[std_jt_mask]                # so now all jet frame cuts have been made (jt and eta)
        std_eta_star = std_eta_star[std_jt_mask]


        ## 3. phi star
        ## remaking 3-vectors
        wta_jet_3v = ak.zip({"px": wta_jets.px, "py": wta_jets.py, "pz": wta_jets.pz}, with_name="Momentum3D")
        wta_constituents_3v = ak.zip({"px": wta_constituents.px, "py": wta_constituents.py, "pz": wta_constituents.pz}, with_name="Momentum3D")
        std_jet_3v = ak.zip({"px": std_jets.px, "py": std_jets.py, "pz": std_jets.pz}, with_name="Momentum3D")
        std_constituents_3v = ak.zip({"px": std_constituents.px, "py": std_constituents.py, "pz": std_constituents.pz}, with_name="Momentum3D")

        z_axis = ak.zip({"px": 0, "py": 0, "pz": 1}, with_name="Momentum3D")

        wta_unit_jets = wta_jet_3v.unit()
        std_unit_jets = std_jet_3v.unit()

        wta_jt_3v = wta_constituents_3v - (wta_unit_jets * wta_unit_jets.dot(wta_constituents_3v)) # 3 vector of jt (i.e. p - unit*(p dot unit))
        std_jt_3v = std_constituents_3v - (std_unit_jets * std_unit_jets.dot(std_constituents_3v))

        wta_phi_origin = wta_unit_jets.cross(wta_unit_jets.cross(z_axis)) #phi = 0 vector
        std_phi_origin = std_unit_jets.cross(std_unit_jets.cross(z_axis))

        ### 1. Normalize your phi origin to use as the X-axis
        wta_x_axis = wta_phi_origin.unit()
        std_x_axis = std_phi_origin.unit()

        ### 2. Create the Y-axis (cross product of Jet and X-axis is automatically a unit vector)
        wta_y_axis = wta_unit_jets.cross(wta_x_axis)    # z hat cross x hat = y hat
        std_y_axis = std_unit_jets.cross(std_x_axis)

        ### 3. Project j_T onto these two axes using the dot product
        wta_proj_x = wta_jt_3v.dot(wta_x_axis)
        wta_proj_y = wta_jt_3v.dot(wta_y_axis)
        std_proj_x = std_jt_3v.dot(std_x_axis)
        std_proj_y = std_jt_3v.dot(std_y_axis)

        ### 4. arctan2 automatically resolves the quadrants to yield -pi to pi
        wta_phi_star = np.arctan2(wta_proj_y, wta_proj_x)
        std_phi_star = np.arctan2(std_proj_y, std_proj_x)

        ### apply theta star and jt cuts (maybe do these earlier to save compute time?)
        wta_phi_star = wta_phi_star[wta_theta_mask][wta_jt_mask]
        std_phi_star = std_phi_star[std_theta_mask][std_jt_mask]

        # consolidating the jet frame values
        wta_energies = wta_constituents.e[wta_theta_mask][wta_jt_mask]
        std_energies = std_constituents.e[std_theta_mask][std_jt_mask]

        wta_jet_frame_particles = ak.zip({
            "pt": wta_jt,
            "eta": wta_eta_star,
            "phi": wta_phi_star,
            "E": wta_energies 
        }, with_name="Momentum4D")

        std_jet_frame_particles = ak.zip({
            "pt": std_jt,
            "eta": std_eta_star,
            "phi": std_phi_star,
            "E": std_energies 
        }, with_name="Momentum4D")

        ## garbage collecting
        del wta_jet_3v, wta_constituents_3v, std_jet_3v, std_constituents_3v
        del z_axis
        del wta_unit_jets, std_unit_jets
        del wta_jt_3v, std_jt_3v
        del wta_phi_origin, std_phi_origin
        del wta_x_axis, std_x_axis, wta_y_axis, std_y_axis, wta_proj_x, wta_proj_y, std_proj_x, std_proj_y
        del wta_jt_mask, std_jt_mask
        del wta_theta_mask, std_theta_mask
        del wta_thetas, std_thetas
        del wta_jt, wta_eta_star, wta_phi_star, wta_energies # the objects are now inside wta_jet_frame_particles ... (?)
        del std_jt, std_eta_star, std_phi_star, std_energies
        gc.collect()


        # sorting the jet frame particles by jet multiplicity
        mult_sorted_jets = {}
        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
            mult_sorted_jets[bin_key] = {
                "wta": wta_jet_frame_particles[bin_pass_dict[bin_key]["wta"]], 
                "std": std_jet_frame_particles[bin_pass_dict[bin_key]["std"]]
            }

        # garbage collecting
        del wta_jet_frame_particles, std_jet_frame_particles
        gc.collect()

        # ----


        # MAKING THE SIGNAL AND EPD (for this chunk) -----
        for bin_key in bin_pass_dict.keys():
            # Filter the jet-frame particles down to just the jets in this bin
            binned_wta_parts = mult_sorted_jets[bin_key]["wta"]
            binned_std_parts = mult_sorted_jets[bin_key]["std"]
            
            # --- FILL EPD FOR CHUNK ---
            wta_epd_eta = ak.to_numpy(ak.flatten(binned_wta_parts.eta, axis=None))
            wta_epd_phi = ak.to_numpy(ak.flatten(binned_wta_parts.phi, axis=None))
            if len(wta_epd_eta) > 0:
                wta_histograms[bin_key]["epd"].FillN(len(wta_epd_eta), wta_epd_eta.astype(np.float64), wta_epd_phi.astype(np.float64), np.ones_like(wta_epd_eta, dtype=np.float64))

            std_epd_eta = ak.to_numpy(ak.flatten(binned_std_parts.eta, axis=None))
            std_epd_phi = ak.to_numpy(ak.flatten(binned_std_parts.phi, axis=None))
            if len(std_epd_eta) > 0:
                std_histograms[bin_key]["epd"].FillN(len(std_epd_eta), std_epd_eta.astype(np.float64), std_epd_phi.astype(np.float64), np.ones_like(std_epd_eta, dtype=np.float64))

            # --- FILL SIGNAL ---
            # 1. WTA signal
            wta_n_sig_pairs = 0
            if num_jets_dict[bin_key]["wta"] > 0:
                # 1. Generate all unique pairs of particles within each individual jet (axis=1)
                wta_pairs = ak.combinations(binned_wta_parts, 2, axis=1) # gives tuple pairs
                if ak.sum(ak.num(wta_pairs)) > 0: # i.e. if more than 0 pairs
                    wta_p1, wta_p2 = ak.unzip(wta_pairs)  # splitting list of (p1, p2) into lists of p1s and then list of p2s
                    # find deta and dphi
                    wta_deta = ak.to_numpy(ak.flatten(abs(wta_p1.eta - wta_p2.eta), axis=None)).astype(np.float64)
                    wta_dphi = ak.to_numpy(ak.flatten(np.arccos(np.cos(wta_p1.phi - wta_p2.phi)), axis=None)).astype(np.float64)
                    
                    # compute the weights
                    wta_n_trig = ak.num(binned_wta_parts, axis=1)
                    # making an array of Ntrig in the shape of wta_p1, i.e. number of associated particles for each trigger:
                    wta_pair_n_trig = ak.broadcast_arrays(wta_n_trig, abs(wta_p1.eta - wta_p2.eta))[0] 
                    wta_weights = ak.to_numpy(ak.flatten(1.0 / wta_pair_n_trig, axis=None)).astype(np.float64)
                    wta_n_sig_pairs = len(wta_deta) #number of signal pairs in this jet

                    # fill the histogram
                    h_sig = wta_histograms[bin_key]["signal"]
                    h_sig.FillN(wta_n_sig_pairs,  wta_deta,  wta_dphi, wta_weights)          
                    h_sig.FillN(wta_n_sig_pairs, -wta_deta,  wta_dphi, wta_weights)          
                    h_sig.FillN(wta_n_sig_pairs,  wta_deta, -wta_dphi, wta_weights)          
                    h_sig.FillN(wta_n_sig_pairs, -wta_deta, -wta_dphi, wta_weights)          
                    h_sig.FillN(wta_n_sig_pairs,  wta_deta, -wta_dphi + 2 * PI, wta_weights) 
                    h_sig.FillN(wta_n_sig_pairs, -wta_deta, -wta_dphi + 2 * PI, wta_weights) 
                    
                    #garbage collection
                    del wta_pairs, wta_p1, wta_p2, wta_deta, wta_dphi, wta_n_trig, wta_pair_n_trig, wta_weights
                    gc.collect()

            # 2. STD signal
            std_n_sig_pairs = 0
            if num_jets_dict[bin_key]["std"] > 0:
                std_pairs = ak.combinations(binned_std_parts, 2, axis=1)
                if ak.sum(ak.num(std_pairs)) > 0:
                    std_p1, std_p2 = ak.unzip(std_pairs)
                    std_deta = ak.to_numpy(ak.flatten(abs(std_p1.eta - std_p2.eta), axis=None)).astype(np.float64)
                    std_dphi = ak.to_numpy(ak.flatten(np.arccos(np.cos(std_p1.phi - std_p2.phi)), axis=None)).astype(np.float64)
                    
                    std_n_trig = ak.num(binned_std_parts, axis=1)
                    std_pair_n_trig = ak.broadcast_arrays(std_n_trig, abs(std_p1.eta - std_p2.eta))[0]
                    std_weights = ak.to_numpy(ak.flatten(1.0 / std_pair_n_trig, axis=None)).astype(np.float64)
                    std_n_sig_pairs = len(std_deta)

                    h_sig = std_histograms[bin_key]["signal"]
                    h_sig.FillN(std_n_sig_pairs,  std_deta,  std_dphi, std_weights)
                    h_sig.FillN(std_n_sig_pairs, -std_deta,  std_dphi, std_weights)
                    h_sig.FillN(std_n_sig_pairs,  std_deta, -std_dphi, std_weights)
                    h_sig.FillN(std_n_sig_pairs, -std_deta, -std_dphi, std_weights)
                    h_sig.FillN(std_n_sig_pairs,  std_deta, -std_dphi + 2 * PI, std_weights)
                    h_sig.FillN(std_n_sig_pairs, -std_deta, -std_dphi + 2 * PI, std_weights)

                    del std_pairs, std_p1, std_p2, std_deta, std_dphi, std_n_trig, std_pair_n_trig, std_weights
                    gc.collect()

            del binned_wta_parts, binned_std_parts, wta_epd_eta, wta_epd_phi, std_epd_eta, std_epd_phi
            gc.collect()

        del mult_sorted_jets, bin_pass_dict
        gc.collect()

    print("All streaming chunks processed. Persistent histograms populated completely.")


    # SAVING THE OUTPUT TO A SINGLE COMBINED COHESIVE ROOT FILE
    os.makedirs(args.outdir, exist_ok=True)
    full_output_path = os.path.join(args.outdir, f"Analysis_Output_Batch{args.batchnum}.root")
    
    print(f"Opening output file to save all results: {full_output_path}")
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
        bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

        # WTA
        hSig_wta = wta_histograms[bin_key]["signal"]
        hEPD_wta = wta_histograms[bin_key]["epd"]
        n_jets_wta = float(num_jets_dict[bin_key]["wta"])
        total_Nch_wta = total_Nch_sum_dict[bin_key]["wta"]

        # STD
        hSig_std = std_histograms[bin_key]["signal"]
        hEPD_std = std_histograms[bin_key]["epd"]
        n_jets_std = float(num_jets_dict[bin_key]["std"])
        total_Nch_std = total_Nch_sum_dict[bin_key]["std"]
        
        # Rename safely inside directory
        hSig_wta.SetName(f"hSig_WTA_{bin_name}")
        hSig_std.SetName(f"hSig_STD_{bin_name}")
        hEPD_wta.SetName(f"hEPD_WTA_{bin_name}")
        hEPD_std.SetName(f"hEPD_STD_{bin_name}")

        # Write out items
        hSig_wta.Write()
        hSig_std.Write()
        hEPD_wta.Write()
        hEPD_std.Write()

        # Write TParameters
        param_total_Nch_wta = ROOT.TParameter('double')(f"total_Nch_WTA_{bin_name}", total_Nch_wta)
        param_total_Nch_std = ROOT.TParameter('double')(f"total_Nch_STD_{bin_name}", total_Nch_std)
        param_total_Nch_wta.Write()
        param_total_Nch_std.Write()

        param_num_jets_wta = ROOT.TParameter('double')(f"num_jets_WTA_{bin_name}", n_jets_wta)
        param_num_jets_std = ROOT.TParameter('double')(f"num_jets_STD_{bin_name}", n_jets_std)
        param_num_jets_wta.Write()
        param_num_jets_std.Write()

        print(f"Written batch {args.batchnum} histograms for multiplicity bin: {bin_name}")

    out_file.Close()
    print(f"Results for batch {args.batchnum} written and saved")

    print("Job successfully completed")
    # END ----


# running:
if __name__ == "__main__":
    main()
