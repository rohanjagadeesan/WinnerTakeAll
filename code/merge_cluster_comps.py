# for merging per-batch outputs from clustertype_comparisons.py

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
    parser = argparse.ArgumentParser(description="Merges parallely generated cluster type comparisons")
    parser.add_argument("--input", required=True, type=str, help="Path to directory of unmerged files")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save merged output file")
    args = parser.parse_args()

    search_pattern = os.path.join(args.input, "Cluster_type_comparison_batch*.root")
    files_to_merge = glob.glob(search_pattern)
    print(f"found {len(files_to_merge)} files to merge")

    # Initialize merged histograms :
    # the difference in raw Nch values:
    merged_blind_Nch_diff_hist = ROOT.TH1D("blind_Nch_diff", "Per-jet difference in N_{ch};N_{ch}^{STD}-N_{ch}^{WTA};Number of jets", 31, -15.5, 15.5)

    # difference in constituents:
    # for particles in standard but not in wta, and vice versa:
    merged_const_unique_to_std = ROOT.TH1D("const_unique_to_std", "Particles in STD but NOT WTA;N_{tracks};Number of Jets", 17, -0.5, 16.5)
    merged_const_unique_to_wta = ROOT.TH1D("const_unique_to_wta", "Particles in WTA but NOT STD;N_{tracks};Number of Jets", 17, -0.5, 16.5)

    # Plain Nch histograms
    merged_wta_Nch_hist = ROOT.TH1D("WTA_Nch", "WTA multiplicity distribution;N_{ch}^{WTA};Number of jets", 130, 0, 130)
    merged_std_Nch_hist = ROOT.TH1D("STD_Nch", "Standard multiplicity distribution;N_{ch}^{STD};Number of jets", 130, 0, 130)

    print("initialised merged histograms")
    
    # iterate through per batch files
    for file_path in files_to_merge:
        # open the file
        tfile = ROOT.TFile.Open(file_path, "READ")

        # add up the blind diff
        blind_Nch_diff = tfile.Get("blind_Nch_diff")
        merged_blind_Nch_diff_hist.Add(blind_Nch_diff)

        # plain Nch histograms
        wta_Nch = tfile.Get("WTA_Nch")
        std_Nch = tfile.Get("STD_Nch")

        merged_wta_Nch_hist.Add(wta_Nch)
        merged_std_Nch_hist.Add(std_Nch)

        #constituent difference histograms
        const_wta = tfile.get("const_unique_to_wta")
        const_std = tfile.Get("const_unique_to_std")

        merged_const_unique_to_std.Add(const_std)
        merged_const_unique_to_wta.Add(const_wta)

        print(f"Added {file_path} to merged histograms")


    # Post-Loop Processing
    print(f"File loop completed - executing post processing")

    # Plot 5: Subtraction (STD - WTA)
    diff_distribution_hist = merged_std_Nch_hist.Clone("STD_minus_WTA_distribution")
    diff_distribution_hist.SetTitle("Multiplicity Distribution Difference (Standard - WTA);N_{ch};#Delta Jets")
    diff_distribution_hist.Add(merged_wta_Nch_hist, -1.0)

    # Plot 6: Ratio (STD / WTA)
    ratio_distribution_hist = merged_std_Nch_hist.Clone("STD_over_WTA_distribution")
    ratio_distribution_hist.SetTitle("Multiplicity Distribution Ratio (Standard / WTA);N_{ch};Ratio")
    ratio_distribution_hist.Divide(merged_wta_Nch_hist)

    # Write out everything to one file
    os.makedirs(args.outdir, exist_ok=True)
    full_output_path = os.path.join(args.outdir, f"Cluster_type_comparison_merged.root")
    
    print(f"Opening output file to save all results: {full_output_path}")
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    merged_blind_Nch_diff_hist.Write()
    merged_const_unique_to_std.Write()
    merged_const_unique_to_wta.Write()
    merged_wta_Nch_hist.Write()
    merged_std_Nch_hist.Write()
    diff_distribution_hist.Write()
    ratio_distribution_hist.Write()
    out_file.Close()

    print(f"Analysis successfully saved to {full_output_path}")
    print("Job completed")

# running:
if __name__ == "__main__":
    main()

