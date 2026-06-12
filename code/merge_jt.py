# for merging jt histograms on nots
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

    parser = argparse.ArgumentParser(description="Merges parallely generated jt files")
    parser.add_argument("--input", required=True, type=str, help="Path to directory of unmerged files")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save merged output file")
    args = parser.parse_args()

    unmerged_dir = os.path.join(args.input)

    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    hists_to_write = []

    ROOT.gDirectory.Clear()

    

    for mult_bin in analysis_bins:
        bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

        
        file_pattern = os.path.join(unmerged_dir, "jt_comparison_batch*.root")
        files = glob.glob(file_pattern)
        summed_wta_hist = None
        summed_std_hist = None

        for file_path in files:
            tfile = ROOT.TFile.Open(file_path, "READ")
            wta_hist_og = tfile.Get(f"WTA_jt_{mult_bin[0]}_{mult_bin[1]}")
            std_hist_og = tfile.Get(f"STD_jt_{mult_bin[0]}_{mult_bin[1]}")
            
            if summed_wta_hist is None: # initialise
                summed_wta_hist = wta_hist_og.Clone()
                summed_wta_hist.SetDirectory(0)
            else:
                summed_wta_hist.Add(wta_hist_og)

            if summed_std_hist is None:
                summed_std_hist = std_hist_og.Clone()
                summed_std_hist.SetDirectory(0)
            else:
                summed_std_hist.Add(std_hist_og)
    
            tfile.Close()
        
        # append the summed histogram to list of hists to write
        hists_to_write.append(summed_wta_hist)
        hists_to_write.append(summed_std_hist)
        print(f"merged {len(files)} files for bin {bin_name}")


    # writing the histograms
    full_output_path = os.path.join(args.outdir, "Merged_jt_comparison.root")
    print(f"Opening output file to save all results: {full_output_path}")
    
    out_file = ROOT.TFile(full_output_path, "RECREATE")

    for hist in hists_to_write:
        hist.Write()

    out_file.Close()

    print(f"Written all histograms to {full_output_path}")
    print("Job completed")


# running:
if __name__ == "__main__":
    main()

