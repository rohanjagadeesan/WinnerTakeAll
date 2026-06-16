# merges and normalises the energy correlators

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
    parser = argparse.ArgumentParser(description="Merges per-batch raw EECs")
    parser.add_argument("--input", required=True, type=str, help="Path to directory of unmerged files") # should be premerge
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save merged EEC file")
    args = parser.parse_args()


    analysis_bins = [ [0,25], [25,36], [36,48], [48,60], [60,71], [71,78], [78,91], [91,97], [97,1000] ]

    ROOT.gDirectory.Clear()


    # INITIALISE MERGED FILES
    merged_wta_histograms = {}
    merged_std_histograms = {}
    num_jets_wta = {}
    num_jets_std = {}

    # Book merged histograms
    ROOT.gDirectory.Clear()
    deltarBW = 0.001
    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
        
        num_jets_wta[bin_key] = 0
        num_jets_std[bin_key] = 0

        merged_wta_histograms[bin_key] = {
            'profile': ROOT.TH1D(f"WTA_profile_{mult_bin[0]}_{mult_bin[1]}", f"WTA Energy profile ({bin_key});#Delta R;Profile" , int(1/deltarBW),0,1),
            'eec': ROOT.TH1D(f"WTA_eec_{mult_bin[0]}_{mult_bin[1]}", f"WTA Energy-Energy Correlator ({bin_key});#Delta R;EEC" ,int(1/deltarBW),0,1)
            }
        
        merged_std_histograms[bin_key] = {
            'profile': ROOT.TH1D(f"STD_profile_{mult_bin[0]}_{mult_bin[1]}", f"Standard Energy profile ({bin_key});#Delta R;Profile" ,int(1/deltarBW),0,1),
            'eec': ROOT.TH1D(f"STD_eec_{mult_bin[0]}_{mult_bin[1]}", f"Standard Energy-Energy Correlator ({bin_key});#Delta R;EEC" ,int(1/deltarBW),0,1)
            }
        
        merged_wta_histograms[bin_key]['profile'].SetDirectory(0)
        merged_std_histograms[bin_key]['profile'].SetDirectory(0)
        merged_wta_histograms[bin_key]['eec'].SetDirectory(0)
        merged_std_histograms[bin_key]['eec'].SetDirectory(0)
        
    print(f"Initialised merged histograms")

    # find files to merge
    search_pattern = os.path.join(args.input, f"EEC_Output_Batch*.root")
    files_to_merge = glob.glob(search_pattern)
    print(f"Found {len(files_to_merge)} files to merge in {args.input}")

    for filepath in files_to_merge:

        file = ROOT.TFile.Open(filepath, "READ")

        for mult_bin in analysis_bins:
            bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"
            bin_name = f"{mult_bin[0]}_{mult_bin[1]}"

            # add the profiles
            wta_profile = file.Get(f"hProfile_WTA_{bin_name}")
            merged_wta_histograms[bin_key]['profile'].Add(wta_profile)

            std_profile = file.Get(f"hProfile_STD_{bin_name}")
            merged_std_histograms[bin_key]['profile'].Add(std_profile)

            # add the EECs
            wta_eec = file.Get(f"hEEC_WTA_{bin_name}")
            merged_wta_histograms[bin_key]['eec'].Add(wta_eec)

            std_eec = file.Get(f"hEEC_STD_{bin_name}")
            merged_std_histograms[bin_key]['eec'].Add(std_eec)

            # add num jets
            njets_wta = file.Get(f"num_jets_WTA_{bin_name}").GetVal()
            njets_std = file.Get(f"num_jets_STD_{bin_name}").GetVal()

            num_jets_wta[bin_key] += njets_wta
            num_jets_std[bin_key] += njets_std

        file.Close()
        print(f"Successfully read {filepath}")


    # Normalise the added histograms
    print("Normalising the merged histograms----")
    normalised_wta_histograms = {}
    normalised_std_histograms = {}
    
    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"

        normalised_wta_histograms[bin_key] = {}
        normalised_std_histograms[bin_key] = {}
        
        # make copies
        normalised_wta_histograms[bin_key]['profile'] = merged_wta_histograms[bin_key]['profile'].Clone()
        normalised_std_histograms[bin_key]['profile'] = merged_std_histograms[bin_key]['profile'].Clone()
        
        normalised_wta_histograms[bin_key]['eec'] = merged_wta_histograms[bin_key]['eec'].Clone()
        normalised_std_histograms[bin_key]['eec'] = merged_std_histograms[bin_key]['eec'].Clone()

        # normalise the histograms 
        if num_jets_wta[bin_key] > 0: # avoiding division by zero
            normalised_wta_histograms[bin_key]['profile'].Scale(1.0 / num_jets_wta[bin_key])
            normalised_wta_histograms[bin_key]['eec'].Scale(1.0 / num_jets_wta[bin_key])
            print(f"{num_jets_wta[bin_key]} WTA jets for {bin_key}")
        else:
            print(f"SKIPPED: 0 wta jets for {bin_key}")
        
        if num_jets_std[bin_key] > 0:
            normalised_std_histograms[bin_key]['profile'].Scale(1.0 / num_jets_std[bin_key])
            normalised_std_histograms[bin_key]['eec'].Scale(1.0 / num_jets_std[bin_key])
            print(f"{num_jets_std[bin_key]} Standard jets for {bin_key}")
        else:
            print(f"SKIPPING: 0 Standard jets for {bin_key}")

    print(f"Finished normalising histograms")


    # save the files
    out_path = os.path.join(args.outdir, "Merged_EECs.root")
    out_file = ROOT.TFile.Open(out_path, "RECREATE")

    print(f"Writing merged signal file to {out_path}")

    for mult_bin in analysis_bins:
        bin_key = f"{mult_bin[0]} < Nch < {mult_bin[1]}"

        normalised_wta_histograms[bin_key]['profile'].Write()
        normalised_std_histograms[bin_key]['profile'].Write()
        normalised_wta_histograms[bin_key]['eec'].Write()
        normalised_std_histograms[bin_key]['eec'].Write()

        print(f"Written normalised merged histograms for bin {mult_bin}")

    out_file.Close()

    print(f"Job completed")

    
# running:
if __name__ == "__main__":
    main()
