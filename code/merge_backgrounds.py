# for merging backgrounds on nots
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

    parser = argparse.ArgumentParser(description="Merges parallely generated backgrounds")
    parser.add_argument("--input", required=True, type=str, help="Path to directory of unmerged files")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save merged output file")
    args = parser.parse_args()

    unmerged_dir = os.path.join(args.input)

    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    hists_to_write = []

    ROOT.gDirectory.Clear()

    y_axis_title = "#frac{1}{N_{trig}} #frac{d^{2}N^{comb}}{d#Delta#phi*#eta*}"

    for mult_bin in analysis_bins:
        bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

        # 1. WTA ----
        # check for split files
        wta_split_pattern = os.path.join(unmerged_dir, f"bkg_WTA_{bin_name}_split*.root")
        wta_split_files = glob.glob(wta_split_pattern)

        if len(wta_split_files) == 0:
            # for not-split files
            wta_path = os.path.join(unmerged_dir, f"WTA_merged_backgrounds_{bin_name}.root")

            # open the wta hist
            wta_tfile = ROOT.TFile.Open(wta_path, "READ")
            wta_hist_og = wta_tfile.Get(f"WTA_bkg_{bin_name}")

            new_wta_hist = wta_hist_og.Clone()
            new_wta_hist.SetTitle(f"WTA background ({mult_bin[0]} < Nch < {mult_bin[1]});#Delta#eta*;#Delta#phi*;{y_axis_title}")
            new_wta_hist.SetDirectory(0)
            hists_to_write.append(new_wta_hist)

            wta_tfile.Close()
        
        else: # i.e. if there are split files
            # collect the split histograms
            summed_wta_hist = None
            for wta_split_path in wta_split_files:
                split_wta_tfile = ROOT.TFile.Open(wta_split_path, "READ")
                split_wta_hist_og = split_wta_tfile.Get(f"WTA_bkg_{bin_name}")
                
                if summed_wta_hist is None: # initialise
                    summed_wta_hist = split_wta_hist_og.Clone()
                    summed_wta_hist.SetDirectory(0)
                    summed_wta_hist.SetTitle(f"WTA background ({mult_bin[0]} < Nch < {mult_bin[1]});#Delta#eta*;#Delta#phi*;{y_axis_title}")
                else:
                    summed_wta_hist.Add(split_wta_hist_og)
                
                split_wta_tfile.Close()
            
            # append the summed histogram to list of hists to write
            hists_to_write.append(summed_wta_hist)
            print(f"merged {len(wta_split_files)} WTA splits for bin {bin_name}")


        # 2. STD ----
        # check for split files
        std_split_pattern = os.path.join(unmerged_dir, f"bkg_STD_{bin_name}_split*.root")
        std_split_files = glob.glob(std_split_pattern)

        if len(std_split_files) == 0:
            # for not-split files
            std_path = os.path.join(unmerged_dir, f"STD_merged_backgrounds_{bin_name}.root")

            # open the wta hist
            std_tfile = ROOT.TFile.Open(std_path, "READ")
            std_hist_og = std_tfile.Get(f"STD_bkg_{bin_name}")

            new_std_hist = std_hist_og.Clone()
            new_std_hist.SetTitle(f"Standard background ({mult_bin[0]} < Nch < {mult_bin[1]});#Delta#eta*;#Delta#phi*;{y_axis_title}")
            new_std_hist.SetDirectory(0)
            hists_to_write.append(new_std_hist)

            std_tfile.Close()
        
        else: # i.e. if there are split files
            # collect the split histograms
            summed_std_hist = None
            for std_split_path in std_split_files:
                split_std_tfile = ROOT.TFile.Open(std_split_path, "READ")
                split_std_hist_og = split_std_tfile.Get(f"STD_bkg_{bin_name}")

                if summed_std_hist is None:
                    # initialising the summed histogram
                    summed_std_hist = split_std_hist_og.Clone()
                    summed_std_hist.SetDirectory(0)
                    summed_std_hist.SetTitle(f"Standard background ({mult_bin[0]} < Nch < {mult_bin[1]});#Delta#eta*;#Delta#phi*;{y_axis_title}")
                else:
                    summed_std_hist.Add(split_std_hist_og)
                
                split_std_tfile.Close()
            
            # add the summed histogram to list of hists to write
            hists_to_write.append(summed_std_hist)
            print(f"merged {len(std_split_files)} STD splits for bin {bin_name}")
        
        
        print(f"Done reading backgrounds for multiplicity bin {mult_bin}")

    # writing the histograms
    full_output_path = os.path.join(args.outdir, "Merged_Backgrounds.root")
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

