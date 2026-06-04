# reads the all-batches merged EPDs and signals and makes the background for each multiplicity bin.

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
    parser = argparse.ArgumentParser(description="Run WTA 2-Particle Correlation on NOTS Cluster")
    parser.add_argument("--input", required=True, type=str, help="Path to the input merged signal & epd file")
    parser.add_argument("--outdir", required=True, type=str, help="Directory where output should be saved")
    args = parser.parse_args()

    

    ROOT.gDirectory.Clear()

    merged_filepath = os.path.join(args.input) 
    merged_file_root = ROOT.TFile.Open(merged_filepath)
    merged_file_up = uproot.open(merged_filepath)

    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    full_output_path = os.path.join(args.outdir, f"merged_backgrounds.root")
    print(f"Output file to save results: {full_output_path}")
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    def fill_6way(hist, n_pairs, deta, dphi, weights):
            PI = np.pi # for 6way fill 
            hist.FillN(n_pairs, (deta).astype(np.float64), (dphi).astype(np.float64), weights)
            hist.FillN(n_pairs, (-deta).astype(np.float64), (dphi).astype(np.float64), weights)
            hist.FillN(n_pairs, (deta).astype(np.float64), (-dphi).astype(np.float64), weights)
            hist.FillN(n_pairs, (-deta).astype(np.float64), (-dphi).astype(np.float64), weights)
            hist.FillN(n_pairs, (deta).astype(np.float64), (-dphi + 2 * PI).astype(np.float64), weights)
            hist.FillN(n_pairs, (-deta).astype(np.float64), (-dphi + 2 * PI).astype(np.float64), weights)



    # MAKING THE BACKGROUNDS
    for mult_bin in analysis_bins:
        bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

        # read signal histograms    
        hSignal_wta = merged_file_root.Get(f"WTA_sig_{bin_name}")
        hSignal_std = merged_file_root.Get(f"STD_sig_{bin_name}")

        # number of signal pairs
        NENT_wta = int(hSignal_wta.GetEntries() // 6)
        NENT_std = int(hSignal_std.GetEntries() // 6)

        print(f"{NENT_wta} WTA signal pairs found for bin {mult_bin}")
        print(f"{NENT_std} STandard signal pairs found for bin {mult_bin}")

        # number of bg pairs should be 10x
        n_bg_pairs_wta = 10 * NENT_wta
        n_bg_pairs_std = 10 * NENT_std

        # initialising background histograms
        hBckrnd_wta = hSignal_wta.Clone(f"WTA_bkg_{bin_name}")
        hBckrnd_std = hSignal_std.Clone(f"STD_bkg_{bin_name}")

        hBckrnd_wta.SetTitle(f"WTA Background {mult_bin[0]} < Nch < {mult_bin[1]}")
        hBckrnd_std.SetTitle(f"Standard Background {mult_bin[0]} < Nch < {mult_bin[1]}")

        hBckrnd_wta.Reset()
        hBckrnd_std.Reset()

        # read EPD coordinates to numpy arrays
        wta_epd = merged_file_up[f"WTA_epd_{bin_name}"]
        std_epd = merged_file_up[f"STD_epd_{bin_name}"]

        # Extract the bin contents and coordinate edges
        wta_counts, wta_eta_edges, wta_phi_edges = wta_epd.to_numpy()
        std_counts, std_eta_edges, std_phi_edges = std_epd.to_numpy()

        # Calculate bin centers and widths
        wta_eta_centers = (wta_eta_edges[:-1] + wta_eta_edges[1:]) / 2
        wta_phi_centers = (wta_phi_edges[:-1] + wta_phi_edges[1:]) / 2
        wta_d_eta = wta_eta_edges[1] - wta_eta_edges[0]
        wta_d_phi = wta_phi_edges[1] - wta_phi_edges[0]

        std_eta_centers = (std_eta_edges[:-1] + std_eta_edges[1:]) / 2
        std_phi_centers = (std_phi_edges[:-1] + std_phi_edges[1:]) / 2
        std_d_eta = std_eta_edges[1] - std_eta_edges[0]
        std_d_phi = std_phi_edges[1] - std_phi_edges[0]

        # 2. Build a 2D coordinate meshgrid corresponding to the bin centers
        wta_eta_grid, wta_phi_grid = np.meshgrid(wta_eta_centers, wta_phi_centers, indexing='ij')
        std_eta_grid, std_phi_grid = np.meshgrid(std_eta_centers, std_phi_centers, indexing='ij')

        # Flatten the grid coordinates and their respective weights
        wta_flat_eta = wta_eta_grid.ravel()
        wta_flat_phi = wta_phi_grid.ravel()
        std_flat_eta = std_eta_grid.ravel()
        std_flat_phi = std_phi_grid.ravel()
        wta_probabilities = wta_counts.ravel()
        std_probabilities = std_counts.ravel()

        # Normalize weights to form a proper probability map
        wta_prob_sum = np.sum(wta_probabilities)
        if wta_prob_sum != 0:
            wta_probabilities /= wta_prob_sum

        std_prob_sum = np.sum(std_probabilities)
        if std_prob_sum > 0:
            std_probabilities /= std_prob_sum

        # 3. Vectorized sampling: pick random bin indices based on EPD weights
        idx1_wta = np.random.choice(len(wta_probabilities), size=n_bg_pairs_wta, p=wta_probabilities)
        idx2_wta = np.random.choice(len(wta_probabilities), size=n_bg_pairs_wta, p=wta_probabilities)

        idx1_std = np.random.choice(len(std_probabilities), size=n_bg_pairs_std, p=std_probabilities)
        idx2_std = np.random.choice(len(std_probabilities), size=n_bg_pairs_std, p=std_probabilities)

        # 4. Un-binning: Add a uniform random smudge within the bin widths
        # This transforms discrete bin centers back into a realistic continuous distribution
        eta1_wta = wta_flat_eta[idx1_wta] + np.random.uniform(-wta_d_eta/2, wta_d_eta/2, size=n_bg_pairs_wta)
        phi1_wta = wta_flat_phi[idx1_wta] + np.random.uniform(-wta_d_phi/2, wta_d_phi/2, size=n_bg_pairs_wta)

        eta2_wta = wta_flat_eta[idx2_wta] + np.random.uniform(-wta_d_eta/2, wta_d_eta/2, size=n_bg_pairs_wta)
        phi2_wta = wta_flat_phi[idx2_wta] + np.random.uniform(-wta_d_phi/2, wta_d_phi/2, size=n_bg_pairs_wta)

        eta1_std = std_flat_eta[idx1_std] + np.random.uniform(-std_d_eta/2, std_d_eta/2, size=n_bg_pairs_std)
        phi1_std = std_flat_phi[idx1_std] + np.random.uniform(-std_d_phi/2, std_d_phi/2, size=n_bg_pairs_std)

        eta2_std = std_flat_eta[idx2_std] + np.random.uniform(-std_d_eta/2, std_d_eta/2, size=n_bg_pairs_std)
        phi2_std = std_flat_phi[idx2_std] + np.random.uniform(-std_d_phi/2, std_d_phi/2, size=n_bg_pairs_std)

        del idx1_wta, idx2_wta, idx1_std, idx2_std
        gc.collect()

        # 5. Compute two-particle kinematics vectorially
        wta_bg_deta = np.abs(eta1_wta - eta2_wta)
        wta_bg_dphi = np.arccos(np.cos(phi1_wta - phi2_wta))

        std_bg_deta = np.abs(eta1_std - eta2_std)
        std_bg_dphi = np.arccos(np.cos(phi1_std - phi2_std))

        del eta1_wta, eta2_wta, phi1_wta, phi2_wta
        del eta1_std, eta2_std, phi1_std, phi2_std
        gc.collect()
        
        # 6. Filling histograms - Explicitly cast arrays to float64 for PyROOT's FillN
        # Also, create a dummy weights array of 1s because TH2::FillN expects (Int_t n, Double_t* x, Double_t* y, Double_t* w)
        wta_weights = np.ones(n_bg_pairs_wta, dtype=np.float64)
        std_weights = np.ones(n_bg_pairs_std, dtype=np.float64)


        fill_6way(hBckrnd_wta, n_bg_pairs_wta, wta_bg_deta, wta_bg_dphi, wta_weights)
        fill_6way(hBckrnd_std, n_bg_pairs_std, std_bg_deta, std_bg_dphi, std_weights)

        hBckrnd_wta.Write()
        hBckrnd_std.Write()
        print(f"Written backgrounds for bin {mult_bin}")

        del hBckrnd_wta, hBckrnd_std
        del wta_bg_deta, wta_bg_dphi, wta_weights, n_bg_pairs_wta
        del std_bg_deta, std_bg_dphi, std_weights, n_bg_pairs_std
        gc.collect()

    out_file.Close()
    print("Job completed")



# running:
if __name__ == "__main__":
    main()
