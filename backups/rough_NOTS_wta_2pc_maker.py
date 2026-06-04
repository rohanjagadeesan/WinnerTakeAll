
# BEFORE RUNNING:
# 1. check input and output (in def saveOutput) locations
# 2. check multiplicity bins 

# maybe change some of the loops to be functions so that garbage collection is handled better ...?

# IMPORTS ---
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


# SETUP ---

# 1. Force ROOT into headless batch mode - i.e. turn off pop up graphics
ROOT.gROOT.SetBatch(True)

# 2. Jet definitions
jet_radius = 0.8 #max radius accepted by fastjet is 1000. radius value in prl paper is 0.8 in lab frame.
wta_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius, fastjet.WTA_pt_scheme) #winner take all definition
std_def = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius) #standard E scheme definition

# INPUT DATA IS OPENED HERE - UPDATE THE PATH
tree = uproot.open("/storage/hpc/work/wl33/ampt_xiao/3mb/nch60_pt500.root:trackTree") # maybe use "with open() as file:""
data = tree.arrays(["genJetPt", "genJetEta", "genDau_pt", "genDau_eta", "genDau_phi", "genDau_chg"]) #opening only the relevant columns
del tree 
gc.collect() # don't need the tree anymore
# ---


# PRELIM ANALYSIS ---

# 1. lab frame jet cuts:
jetEtaCut = 1.6 #from PRL paper page 2
jetPtCut = 550  #from prl paper. This is the STANDARD pt, not wta pt.
data = data[(data.genJetPt > jetPtCut) & (abs(data.genJetEta) < jetEtaCut)] #applying the cuts
data = data[ak.num(data.genJetPt) > 0]

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

analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

bin_pass_dict = {}  # dict of true/false masks for whether jets fall in each bin
num_jets_dict = {}  # dict for number of jets in each bin

for mult_bin in analysis_bins:
    bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
    
    wta_bin_mask = (wta_Nch >= mult_bin[0]) & (wta_Nch < mult_bin[1])
    std_bin_mask = (std_Nch >= mult_bin[0]) & (std_Nch < mult_bin[1])
    
    bin_pass_dict[bin_key] = {"wta" : wta_bin_mask , "std" : std_bin_mask}
    num_jets_dict[bin_key] = {"wta" : ak.sum(wta_bin_mask), "std" : ak.sum(std_bin_mask)} # summing the trues and falses gives number of passing jets per bin


avg_Nch_dict = {} #finding average Nch for each bin

for bin_key, masks in bin_pass_dict.items():
    # 1. Extract the masks for this specific bin
    wta_mask = masks["wta"]
    std_mask = masks["std"]
    
    # 2. Apply the mask to the overall Nch arrays, then take the mean
    # ak.mean automatically sums the tracks and divides by the number of jets in the bin
    wta_avg = ak.mean(wta_Nch[wta_mask])
    std_avg = ak.mean(std_Nch[std_mask])
    
    # 3. Store the results
    avg_Nch_dict[bin_key] = {
        "wta": wta_avg,
        "std": std_avg
    }

# garbage collecting:
del wta_avg, std_avg, wta_mask, std_mask
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
wta_phi_star = wta_phi_star[wta_theta_mask]
std_phi_star = std_phi_star[std_theta_mask]
wta_phi_star = wta_phi_star[wta_jt_mask]
std_phi_star = std_phi_star[std_jt_mask]

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
# maybe del wta_jt, wta_eta_star, wta_phi_star, wta_energies ? the objects are now inside wta_jet_frame_particles ... 
# same for std?
gc.collect()


# sorting the jet frame particles by jet multiplicity
mult_sorted_jets = {}
for mult_bin in analysis_bins:
    bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
    
    wta_bin_mask = bin_pass_dict[bin_key]["wta"]
    std_bin_mask = bin_pass_dict[bin_key]["std"]
    
    wta_sorted_particles = wta_jet_frame_particles[wta_bin_mask]
    std_sorted_particles = std_jet_frame_particles[std_bin_mask]

    mult_sorted_jets[bin_key] = {"wta": wta_sorted_particles, "std": std_sorted_particles}

#garbage collecting
del bin_key, wta_bin_mask, std_bin_mask, wta_jet_frame_particles, std_jet_frame_particles
del wta_sorted_particles, std_sorted_particles
gc.collect()

# ----


# MAKING THE SIGNAL -----
wta_signal_results = {} # Dictionaries to store the final flat signal arrays for each multiplicity bin
std_signal_results = {}

for bin_key, masks in bin_pass_dict.items():
    #print(f"Processing signal pairs for bin: {bin_key}...")
    
    # Filter the jet-frame particles down to just the jets in this bin
    binned_wta_parts = mult_sorted_jets[bin_key]["wta"]
    binned_std_parts = mult_sorted_jets[bin_key]["std"]
    
    # WTA FRAME SIGNAL PAIRS
    if num_jets_dict[bin_key]["wta"] > 0: # i.e. not empty
        # 1. Generate all unique pairs of particles within each individual jet (axis=1)
        wta_pairs = ak.combinations(binned_wta_parts, 2, axis=1) # gives tuples of particle pairs
        
        if ak.sum(ak.num(wta_pairs)) > 0: # i.e. if there are more than zero pairs,
            
            wta_p1, wta_p2 = ak.unzip(wta_pairs)  # separates into 2 arrays, one for each half of the pairs
            # 2. Compute delta_eta* and delta_phi*
            wta_deta = abs(wta_p1.eta - wta_p2.eta)
            wta_dphi = np.arccos(np.cos(wta_p1.phi - wta_p2.phi))
            
            # 4. Compute weights : 1/Ntrig , Ntrig is the number of trigger particles in the jet
            # use energies for EEC
            # 1. Calculate N_trig for each jet in the layout
            wta_n_trig = ak.num(binned_wta_parts, axis=1)# (This counts particles inside each sub-list before pairs were made)
            # 2. Broadcast N_trig from 1 per jet to 1 per pair using ak.broadcast_arrays
            wta_pair_n_trig = ak.broadcast_arrays(wta_n_trig, wta_deta)[0]
            # 3. Calculate weight = 1 / N_trig
            wta_weights = 1.0 / wta_pair_n_trig

            # 5. Flatten to 1D arrays for easy direct filling into ROOT TH2D histograms
            wta_signal_results[bin_key] = {
                "deta": ak.flatten(wta_deta, axis=None),
                "dphi": ak.flatten(wta_dphi, axis=None),
                "weight": ak.flatten(wta_weights, axis=None)
            }
    
            #garbage collection
            del wta_pairs, wta_p1, wta_p2
            del wta_deta, wta_dphi
            del wta_weights, wta_n_trig, wta_pair_n_trig
            gc.collect()
        

    # STANDARD FRAME SIGNAL PAIRS
    if num_jets_dict[bin_key]["std"] > 0:
        # 1. Generate unique pairs
        std_pairs = ak.combinations(binned_std_parts, 2, axis=1)
        
        if ak.sum(ak.num(std_pairs)) > 0:
            std_p1, std_p2 = ak.unzip(std_pairs)
            
            # 2. Compute delta_eta* and delta_phi*
            std_deta = abs(std_p1.eta - std_p2.eta)
            std_dphi = np.arccos(np.cos(std_p1.phi - std_p2.phi))
            
            # 4. Compute weights
            std_n_trig = ak.num(binned_std_parts, axis=1)
            std_pair_n_trig = ak.broadcast_arrays(std_n_trig, std_deta)[0]
            std_weights = 1.0 / std_pair_n_trig
            
            # 5. Flatten to 1D arrays
            std_signal_results[bin_key] = {
                "deta": ak.flatten(std_deta, axis=None),
                "dphi": ak.flatten(std_dphi, axis=None),
                "weight": ak.flatten(std_weights, axis=None)
            }
    
            # garbage collecting
            del std_n_trig, std_pair_n_trig, std_weights
            del std_deta, std_dphi
            del std_pairs, std_p1, std_p2
            gc.collect()

# signal pairs now completely generated

# garbage collecting:
del binned_wta_parts, binned_std_parts
gc.collect()

# ----

# MAKING THE EPD: ----
wta_epd_results = {}
std_epd_results = {}

for bin_key in bin_pass_dict.keys():
    
    #print(f"Processing EPD for bin: {bin_key}...")
    
    # Filter the jet-frame particles down to just the jets in this bin
    binned_wta_parts = mult_sorted_jets[bin_key]["wta"]
    binned_std_parts = mult_sorted_jets[bin_key]["std"]
    
    # CREATE THE EPD (SINGLE PARTICLE DISTRIBUTION POOLS)
    # Flatten all particles across all jets in this bin to make a 1D pool
    wta_epd_eta = ak.to_numpy(ak.flatten(binned_wta_parts.eta, axis=None))
    wta_epd_phi = ak.to_numpy(ak.flatten(binned_wta_parts.phi, axis=None))
    wta_epd_results[bin_key] = {"eta": wta_epd_eta, "phi": wta_epd_phi}
    
    std_epd_eta = ak.to_numpy(ak.flatten(binned_std_parts.eta, axis=None))
    std_epd_phi = ak.to_numpy(ak.flatten(binned_std_parts.phi, axis=None))
    std_epd_results[bin_key] = {"eta": std_epd_eta, "phi": std_epd_phi}

# EPD generation complete
# garbage collection:
del binned_wta_parts, binned_std_parts
del wta_epd_eta, wta_epd_phi, std_epd_eta, std_epd_phi
gc.collect()
# -----

# GENERATING THE BACKGROUND ----
wta_background_results = {}
std_background_results = {}

#maybe np.random.seed for reproducability ...?

for bin_key in bin_pass_dict.keys():
    #print(f"Processing background for bin: {bin_key}...")
    
    # Filter the jet-frame particles down to just the jets in this bin
    binned_wta_parts = mult_sorted_jets[bin_key]["wta"]
    binned_std_parts = mult_sorted_jets[bin_key]["std"]

    wta_epd_eta = wta_epd_results[bin_key]["eta"]
    wta_epd_phi = wta_epd_results[bin_key]["phi"]
    std_epd_eta = std_epd_results[bin_key]["eta"]
    std_epd_phi = std_epd_results[bin_key]["phi"]
    
    # 2. WTA BACKGROUND SAMPLING
    if bin_key in wta_signal_results:
        # Determine the target number of background pairs (10x Signal)
        n_wta_sig_pairs = len(wta_signal_results[bin_key]["deta"])
        n_wta_bg_pairs = 10 * n_wta_sig_pairs
        
        #print(f"Sampling {n_wta_bg_pairs} WTA background pairs for {bin_key}...")
        
        # Draw random indices from the EPD pool
        idx1 = np.random.randint(0, len(wta_epd_eta), size=n_wta_bg_pairs)
        idx2 = np.random.randint(0, len(wta_epd_eta), size=n_wta_bg_pairs)
            
        # Compute absolute differences
        wta_bg_deta = abs(wta_epd_eta[idx1] - wta_epd_eta[idx2])
        wta_bg_dphi = np.arccos(np.cos(wta_epd_phi[idx1] - wta_epd_phi[idx2]))
        
        wta_background_results[bin_key] = {
            "deta": wta_bg_deta,
            "dphi": wta_bg_dphi,
            "weight": np.ones_like(wta_bg_deta)
        }

        # make sure inside loop garbage collection is fine ...
        del wta_bg_deta, wta_bg_dphi
        del idx1, idx2
        del n_wta_bg_pairs, n_wta_sig_pairs
        gc.collect()

    # 3. STANDARD BACKGROUND SAMPLING
    if bin_key in std_signal_results:
        n_std_sig_pairs = len(std_signal_results[bin_key]["deta"])
        n_std_bg_pairs = 10 * n_std_sig_pairs
        
        #print(f"Sampling {n_std_bg_pairs} STD background pairs for {bin_key}...")
        
        idx1 = np.random.randint(0, len(std_epd_eta), size=n_std_bg_pairs)
        idx2 = np.random.randint(0, len(std_epd_eta), size=n_std_bg_pairs)
            
        std_bg_deta = abs(std_epd_eta[idx1] - std_epd_eta[idx2])
        std_bg_dphi = np.arccos(np.cos(std_epd_phi[idx1] - std_epd_phi[idx2]))
        
        std_background_results[bin_key] = {
            "deta": std_bg_deta,
            "dphi": std_bg_dphi,
            "weight": np.ones_like(std_bg_deta)
        }

        # garbage collecting
        del idx1, idx2
        del n_std_bg_pairs, n_std_sig_pairs
        del std_bg_deta, std_bg_dphi
        gc.collect()

# background generation complete

# some more garbage collecting:
del binned_wta_parts, binned_std_parts
del wta_epd_eta, wta_epd_phi, std_epd_eta, std_epd_phi

# -----


# POPULATE HISTOGRAMS ----
ROOT.gDirectory.Clear() # just in case, to avoid memory leakage and errors

# Spatial Constants
PI = np.pi

# Dictionaries to store the booked ROOT objects
wta_histograms = {}
std_histograms = {}

# eta and phi bins
eta_bins, eta_min, eta_max = 41, -6.15, 6.15    # from xiao's code. prl paper uses -3 to 3
phi_bins = 33
phi_min = -(math.pi/2.0) + (math.pi/32.0)
phi_max = (3*math.pi/2.0) + (math.pi/32.0)
phi_bin_width = (phi_max - phi_min)/phi_bins

for bin_key in bin_pass_dict.keys():
    # Sanitize the multiplicity string so it complies with ROOT naming properties
    safe_name = bin_key.replace(" ", "_").replace("<", "lt").replace(">", "gt")
   
    # 1. INITIALISE HISTOGRAMS (TH2D)
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


    # 2. FILL EPD HISTOGRAMS
    if bin_key in wta_epd_results:
        wta_epd = wta_epd_results[bin_key]
        n_epd = len(wta_epd["eta"])
        if n_epd > 0:
            wta_histograms[bin_key]["epd"].FillN(
                n_epd, 
                ak.to_numpy(wta_epd["eta"]).astype(np.float64), 
                ak.to_numpy(wta_epd["phi"]).astype(np.float64), 
                np.ones(n_epd, dtype=np.float64)
            )
            
    if bin_key in std_epd_results:
        std_epd = std_epd_results[bin_key]
        n_epd = len(std_epd["eta"])
        if n_epd > 0:
            std_histograms[bin_key]["epd"].FillN(
                n_epd, 
                ak.to_numpy(std_epd["eta"]).astype(np.float64), 
                ak.to_numpy(std_epd["phi"]).astype(np.float64), 
                np.ones(n_epd, dtype=np.float64)
            )


    # 3. FILL CORRELATION HISTOGRAMS 
    # --- WTA Frame ---
    if bin_key in wta_signal_results:
        sig = wta_signal_results[bin_key]
        n_sig = len(sig["deta"])
        if n_sig > 0:
            # FIX: Convert Awkward Arrays to NumPy before casting type
            deta = ak.to_numpy(sig["deta"]).astype(np.float64)
            dphi = ak.to_numpy(sig["dphi"]).astype(np.float64)
            weights = ak.to_numpy(sig["weight"]).astype(np.float64)
            h_sig = wta_histograms[bin_key]["signal"]
            
            h_sig.FillN(n_sig,  deta,  dphi, weights)          # 1. (+deta, +dphi)
            h_sig.FillN(n_sig, -deta,  dphi, weights)          # 2. (-deta, +dphi)
            h_sig.FillN(n_sig,  deta, -dphi, weights)          # 3. (+deta, -dphi)
            h_sig.FillN(n_sig, -deta, -dphi, weights)          # 4. (-deta, -dphi)
            h_sig.FillN(n_sig,  deta, -dphi + 2 * PI, weights) # 5. (+deta, -dphi + 2pi)
            h_sig.FillN(n_sig, -deta, -dphi + 2 * PI, weights) # 6. (-deta, -dphi + 2pi)

    if bin_key in wta_background_results:
        bg = wta_background_results[bin_key]
        n_bg = len(bg["deta"])
        if n_bg > 0:
            # FIX: Convert to NumPy array safely
            deta = ak.to_numpy(bg["deta"]).astype(np.float64)
            dphi = ak.to_numpy(bg["dphi"]).astype(np.float64)
            weights = ak.to_numpy(bg["weight"]).astype(np.float64)
            h_bg = wta_histograms[bin_key]["background"]
            
            h_bg.FillN(n_bg,  deta,  dphi, weights)
            h_bg.FillN(n_bg, -deta,  dphi, weights)
            h_bg.FillN(n_bg,  deta, -dphi, weights)
            h_bg.FillN(n_bg, -deta, -dphi, weights)
            h_bg.FillN(n_bg,  deta, -dphi + 2 * PI, weights)
            h_bg.FillN(n_bg, -deta, -dphi + 2 * PI, weights)

    # --- Standard Frame ---
    if bin_key in std_signal_results:
        sig = std_signal_results[bin_key]
        n_sig = len(sig["deta"])
        if n_sig > 0:
            # FIX: Convert to NumPy array safely
            deta = ak.to_numpy(sig["deta"]).astype(np.float64)
            dphi = ak.to_numpy(sig["dphi"]).astype(np.float64)
            weights = ak.to_numpy(sig["weight"]).astype(np.float64)
            h_sig = std_histograms[bin_key]["signal"]
            
            h_sig.FillN(n_sig,  deta,  dphi, weights)
            h_sig.FillN(n_sig, -deta,  dphi, weights)
            h_sig.FillN(n_sig,  deta, -dphi, weights)
            h_sig.FillN(n_sig, -deta, -dphi, weights)
            h_sig.FillN(n_sig,  deta, -dphi + 2 * PI, weights)
            h_sig.FillN(n_sig, -deta, -dphi + 2 * PI, weights)

    if bin_key in std_background_results:
        bg = std_background_results[bin_key]
        n_bg = len(bg["deta"])
        if n_bg > 0:
            # FIX: Convert to NumPy array safely
            deta = ak.to_numpy(bg["deta"]).astype(np.float64)
            dphi = ak.to_numpy(bg["dphi"]).astype(np.float64)
            weights = ak.to_numpy(bg["weight"]).astype(np.float64)
            h_bg = std_histograms[bin_key]["background"]
            
            h_bg.FillN(n_bg,  deta,  dphi, weights)
            h_bg.FillN(n_bg, -deta,  dphi, weights)
            h_bg.FillN(n_bg,  deta, -dphi, weights)
            h_bg.FillN(n_bg, -deta, -dphi, weights)
            h_bg.FillN(n_bg,  deta, -dphi + 2 * PI, weights)
            h_bg.FillN(n_bg, -deta, -dphi + 2 * PI, weights)

#All Signal, Background, and EPD ROOT histograms populated completely

# garbage collection?

# -----


# GENERATE YIELDS ----
wta_yields = {} # Dictionaries for yield histograms
std_yields = {}
y_axis_title = "#frac{1}{N_{trig}} #frac{d^{2}N}{d#Delta#phi*#eta*}"

for bin_key in bin_pass_dict.keys():
    # Sanitize the bin key string for valid ROOT naming conventions
    safe_name = bin_key.replace(" ", "_").replace("<", "lt").replace(">", "gt")
    
    # A. WTA FRAME YIELD
    n_jets_wta = float(num_jets_dict[bin_key]["wta"])
    h_sig_wta = wta_histograms[bin_key]["signal"]
    h_bg_wta = wta_histograms[bin_key]["background"]
    h_epd_wta = wta_histograms[bin_key]["epd"]
    
    if n_jets_wta > 0 and h_sig_wta.GetEntries() > 0 and h_bg_wta.GetEntries() > 0:
        # 1. Clone the signal histogram to preserve binning and layout
        h_yield_wta = h_sig_wta.Clone(f"WTA_yield_{safe_name}")
        h_yield_wta.SetTitle(f"WTA Yield ({bin_key});#Delta#eta*;#Delta#phi*;{y_axis_title}")
        
        # 2. Extract B(0,0) by querying the bin coordinates containing (0,0)
        bin_zero_wta = h_bg_wta.FindBin(0.0, 0.0)
        b00_wta = h_bg_wta.GetBinContent(bin_zero_wta)
        
        # 3. Perform bin-by-bin division: S(delta_eta, delta_phi) / B(delta_eta, delta_phi)
        h_yield_wta.Divide(h_bg_wta)
        
        # 4. Scale by B(0,0) / N_jet
        if b00_wta > 0:
            h_yield_wta.Scale(b00_wta / n_jets_wta)
            
        wta_yields[bin_key] = h_yield_wta
        

    # B. STANDARD FRAME YIELD
    n_jets_std = float(num_jets_dict[bin_key]["std"])
    h_sig_std = std_histograms[bin_key]["signal"]
    h_bg_std = std_histograms[bin_key]["background"]
    h_epd_std = std_histograms[bin_key]["epd"]
    
    if n_jets_std > 0 and h_sig_std.GetEntries() > 0 and h_bg_std.GetEntries() > 0:
        # 1. Clone the standard frame signal histogram
        h_yield_std = h_sig_std.Clone(f"h2_std_yield_{safe_name}")
        h_yield_std.SetTitle(f"Standard Yield ({bin_key});#Delta#eta*;#Delta#phi*;{y_axis_title}")
        
        # 2. Extract B(0,0)
        bin_zero_std = h_bg_std.FindBin(0.0, 0.0)
        b00_std = h_bg_std.GetBinContent(bin_zero_std)
        
        # 3. Perform bin-by-bin division
        h_yield_std.Divide(h_bg_std)
        
        # 4. Scale by B(0,0) / N_jet
        if b00_std > 0:
            h_yield_std.Scale(b00_std / n_jets_std)
            
        std_yields[bin_key] = h_yield_std
        

# Normalised yield histograms are filled in

# garbage collection?

# ----


# SAVING THE OUTPUT

# defining the function
def saveOutput(mult_bin, hYield_wta, hSig_wta, hBkg_wta, hEPD_wta, avg_Nch_wta, hYield_std, hSig_std, hBkg_std, hEPD_std, avg_Nch_std):
    '''
    saves the 2d histograms
    '''
    # 1. Define your specific output folder path
    output_dir = "/home/rj65/winner_take_all/output"

    # Create the folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Create the dynamic filename
    bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

    filename = f"Yield_Histograms_Mult_{bin_name}.root"

    # 3. Combine the folder path and the filename
    full_filepath = os.path.join(output_dir, filename)

    # 4. Open the ROOT file using the FULL path
    out_file = ROOT.TFile(full_filepath, "RECREATE")

    # 5. Rename and write the histograms
    hYield_wta.SetName(f"hYield_WTA_{bin_name}")
    hYield_std.SetName(f"hYield_STD_{bin_name}")
    hSig_wta.SetName(f"hSig_WTA_{bin_name}")
    hSig_std.SetName(f"hSig_STD_{bin_name}")
    hBkg_wta.SetName(f"hBkg_WTA_{bin_name}")
    hBkg_std.SetName(f"hBkg_STD_{bin_name}")
    hEPD_wta.SetName(f"hEPD_WTA_{bin_name}")
    hEPD_std.SetName(f"hEPD_STD_{bin_name}")

    hYield_wta.Write()
    hYield_std.Write()
    hSig_wta.Write()
    hSig_std.Write()
    hBkg_wta.Write()
    hBkg_std.Write()
    hEPD_wta.Write()
    hEPD_std.Write()

    # 6. Create and Write the TParameters for avg Nch per jet
    param_avg_Nch_wta = ROOT.TParameter('double')("avg_Nch_WTA", avg_Nch_wta)
    param_avg_Nch_std = ROOT.TParameter('double')("avg_Nch_STD", avg_Nch_std)
    param_avg_Nch_wta.Write()
    param_avg_Nch_std.Write()

    out_file.Close()


# running the function
for mult_bin in analysis_bins:
    bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"

    if (bin_key in std_yields.keys()) and (bin_key in wta_yields.keys()):
        hYield_wta = wta_yields[bin_key]
        hSig_wta = wta_histograms[bin_key]["signal"]
        hBkg_wta = wta_histograms[bin_key]["background"]
        hEPD_wta = wta_histograms[bin_key]["epd"]
        avg_Nch_wta = avg_Nch_dict[bin_key]["wta"]

        hYield_std = std_yields[bin_key]
        hSig_std = std_histograms[bin_key]["signal"]
        hBkg_std = std_histograms[bin_key]["background"]
        hEPD_std = std_histograms[bin_key]["epd"]
        avg_Nch_std = avg_Nch_dict[bin_key]["std"]

        saveOutput(mult_bin, hYield_wta, hSig_wta, hBkg_wta, hEPD_wta, avg_Nch_wta, hYield_std, hSig_std, hBkg_std, hEPD_std, avg_Nch_std)

# final output histograms have now been saved

# garbage collection?

# END ----