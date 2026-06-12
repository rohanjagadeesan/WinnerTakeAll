# merge the per-batch signals and EPDs

# IMPORTS---
import ROOT
import numpy as np
import math
import os
import argparse
import glob
# ---


def main():

    # parse the inputs
    parser = argparse.ArgumentParser(description="Merges parallely generated Signals and EPDs")
    parser.add_argument("--input", required=True, type=str, help="Path to directory of unmerged files") # should be premerge
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save merged signal & EPD file")
    args = parser.parse_args()


    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    ROOT.gDirectory.Clear()
    y_axis_title = "#frac{1}{N_{trig}} #frac{d^{2}N^{sig}}{d#Delta#phi*#eta*}"


    # INITIALISE MERGED FILES
    merged_wta_sig_hists = {}
    merged_std_sig_hists = {}
    
    merged_wta_epd_hists = {}
    merged_std_epd_hists = {}

    merged_wta_total_Nch = {}
    merged_std_total_Nch = {}

    merged_wta_num_jets = {}
    merged_std_num_jets = {}

    merged_wta_avg_Nch = {}
    merged_std_avg_Nch = {}

    # eta and phi bins
    eta_bins, eta_min, eta_max = 41, -6.15, 6.15    # from xiao's code. prl paper uses -3 to 3
    phi_bins = 33
    phi_min = -(math.pi/2.0) + (math.pi/32.0)
    phi_max = (3*math.pi/2.0) + (math.pi/32.0)
    phi_bin_width = (phi_max - phi_min)/phi_bins

    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} <  Nch < {mult_bin[1]}"
        safe_name = f"{mult_bin[0]}_{mult_bin[1]}"
    
        # 1. INITIALISE HISTOGRAMS (TH2D)
        merged_wta_sig_hists[bin_key] = ROOT.TH2D(f"WTA_sig_{safe_name}", f"WTA Signal ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max)
        merged_std_sig_hists[bin_key] = ROOT.TH2D(f"STD_sig_{safe_name}", f"Standard Signal ({bin_key});#Delta#eta*;#Delta#phi*", eta_bins, eta_min, eta_max, phi_bins, phi_min, phi_max)

        merged_wta_sig_hists[bin_key].SetTitle( f"WTA signal ({bin_key});#Delta#eta*;#Delta#phi*;{y_axis_title}")
        merged_std_sig_hists[bin_key].SetTitle( f"Standard signal ({bin_key});#Delta#eta*;#Delta#phi*;{y_axis_title}")

        merged_wta_epd_hists[bin_key] = ROOT.TH2D(f"WTA_epd_{safe_name}", f"WTA EPD ({bin_key});#eta*;#phi*", 150, 0, 10, 120, -4, 4)
        merged_std_epd_hists[bin_key] = ROOT.TH2D(f"STD_epd_{safe_name}", f"Standard EPD ({bin_key});#eta*;#phi*", 150, 0, 10, 120, -4, 4)

        # initialise param values
        merged_wta_total_Nch[bin_key] = 0
        merged_std_total_Nch[bin_key] = 0

        merged_wta_num_jets[bin_key] = 0
        merged_std_num_jets[bin_key] = 0

        #merged_wta_avg_Nch[bin_key] = 0
        #merged_std_avg_Nch[bin_key] = 0
    
    print(f"Initialised merged histograms and parameters")

    # find files to merge
    search_pattern = os.path.join(args.input, f"Analysis_Output_Batch*.root")
    files_to_merge = glob.glob(search_pattern)
    print(f"Found {len(files_to_merge)} files to merge in {args.input}")

    for filepath in files_to_merge:

        file = ROOT.TFile.Open(filepath, "READ")

        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} <  Nch < {mult_bin[1]}"

            # open histograms
            hSig_wta = file.Get(f"hSig_WTA_{mult_bin[0]}_{mult_bin[1]}")
            hSig_std = file.Get(f"hSig_STD_{mult_bin[0]}_{mult_bin[1]}")

            hEPD_wta = file.Get(f"hEPD_WTA_{mult_bin[0]}_{mult_bin[1]}")
            hEPD_std = file.Get(f"hEPD_STD_{mult_bin[0]}_{mult_bin[1]}")

            # add them to the merged histograms
            merged_wta_sig_hists[bin_key].Add(hSig_wta)
            merged_std_sig_hists[bin_key].Add(hSig_std)

            merged_wta_epd_hists[bin_key].Add(hEPD_wta)
            merged_std_epd_hists[bin_key].Add(hEPD_std)

            # read parameters
            total_Nch_wta = file.Get(f"total_Nch_WTA_{mult_bin[0]}_{mult_bin[1]}").GetVal()
            total_Nch_std = file.Get(f"total_Nch_STD_{mult_bin[0]}_{mult_bin[1]}").GetVal()

            num_jets_wta = file.Get(f"num_jets_WTA_{mult_bin[0]}_{mult_bin[1]}").GetVal()
            num_jets_std = file.Get(f"num_jets_STD_{mult_bin[0]}_{mult_bin[1]}").GetVal()

            # add them to the dictionaries
            merged_wta_num_jets[bin_key] += num_jets_wta
            merged_std_num_jets[bin_key] += num_jets_std

            merged_wta_total_Nch[bin_key] += total_Nch_wta
            merged_std_total_Nch[bin_key] += total_Nch_std

            # avg Nch (will only be accurate at the end)
            if merged_wta_num_jets[bin_key] > 0:
                merged_wta_avg_Nch[bin_key] = merged_wta_total_Nch[bin_key] / merged_wta_num_jets[bin_key] 
            else:
                merged_wta_avg_Nch[bin_key] = 0
            
            if merged_std_num_jets[bin_key] > 0:
                merged_std_avg_Nch[bin_key] = merged_std_total_Nch[bin_key] / merged_std_num_jets[bin_key]
            else:
                merged_std_avg_Nch[bin_key] = 0
                

        file.Close()
        print(f"Successfully read {filepath}")

    
    # save the files
    out_path = os.path.join(args.outdir, "Merged_Signals.root")
    out_file = ROOT.TFile.Open(out_path, "RECREATE")

    print(f"Writing merged signal file to {out_path}")

    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} <  Nch < {mult_bin[1]}"
        
        # write signals to file
        merged_wta_sig_hists[bin_key].Write()
        merged_std_sig_hists[bin_key].Write()

        # write EPDs
        merged_wta_epd_hists[bin_key].Write()
        merged_std_epd_hists[bin_key].Write()

        # write avg Nch
        param_avg_Nch_wta = ROOT.TParameter('double')(f"avg_Nch_WTA_{mult_bin[0]}_{mult_bin[1]}", merged_wta_avg_Nch[bin_key])
        param_avg_Nch_std = ROOT.TParameter('double')(f"avg_Nch_STD_{mult_bin[0]}_{mult_bin[1]}", merged_std_avg_Nch[bin_key])
        param_avg_Nch_wta.Write()
        param_avg_Nch_std.Write()

        # write number of jets
        param_num_jets_wta = ROOT.TParameter('double')(f"num_jets_WTA_{mult_bin[0]}_{mult_bin[1]}", merged_wta_num_jets[bin_key])
        param_num_jets_std = ROOT.TParameter('double')(f"num_jets_STD_{mult_bin[0]}_{mult_bin[1]}", merged_std_num_jets[bin_key])
        param_num_jets_wta.Write()
        param_num_jets_std.Write()

        # write total Nch
        param_total_Nch_wta = ROOT.TParameter('double')(f"total_Nch_WTA_{mult_bin[0]}_{mult_bin[1]}", merged_wta_total_Nch[bin_key])
        param_total_Nch_std = ROOT.TParameter('double')(f"total_Nch_STD_{mult_bin[0]}_{mult_bin[1]}", merged_std_total_Nch[bin_key])
        param_total_Nch_wta.Write()
        param_total_Nch_std.Write()

        print(f"Written merged parameters and histograms for bin {mult_bin}")

    out_file.Close()

    print(f"Job completed")

    
# running:
if __name__ == "__main__":
    main()
