"""
This script plots the features for different datasets, 
in order to be able to compare them.
"""
import uproot as ur
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

# Matplotlib settings
params = {
    'font.size': 13
}
plt.rcParams.update(params)


def apply_cuts(df):
    """ Apply some cuts on the variables based on their physical meaning"""
    df = df[df["clusterE"]>0.]
    df = df[df["cluster_CENTER_LAMBDA"]>0.]
    df = df[df["cluster_FIRST_ENG_DENS"]>0.]
    df = df[df["cluster_SECOND_TIME"]>0.]


def main():

    ################
    #  Read files  #
    ################ 

    # Import files
    file_bkg_old = ur.open('/eos/user/g/gfazzino/pileupdata/fromChris/moreStats_mltree_mc20e_withPileup.root') 
    file_sig_old = ur.open('/eos/user/g/gfazzino/pileupdata/fromChris/moreStats_mltree_mc20e_noPileup.root') 
    file_bkg_new = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc20e/mc20e_withPU.root') 
    file_sig_new = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc20e/mc20e_noPU.root') 
    #file_bkg_new = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc23d/mc23d_withPU.root') 
    #file_sig_new = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc23d/mc23d_noPU.root')
    print('Found files, reading datasets... \n')

    # Output paths
    out_path = 'out'
    fig_path = out_path +'/plots'

    # Create (if necessary) output folders
    try:
        os.system("mkdir {}".format(out_path))
    except ImportError:
        print("{} already exists \n".format(out_path))
    pass

    try:
        os.system("mkdir {}".format(fig_path))
    except ImportError:
        print("{} already exists \n".format(fig_path))
    pass 

    print('Reading datasets complete \n')

    ###########################
    #  Dataframe Preparation  #
    ########################### 

    # Make pandas dataframes
    tree_bkg_old = file_bkg_old['ClusterTree']
    df_bkg_old = tree_bkg_old.arrays(library='pd')

    tree_sig_old = file_sig_old['ClusterTree']
    df_sig_old = tree_sig_old.arrays(library='pd')

    tree_bkg_new = file_bkg_new['ClusterTree']
    df_bkg_new = tree_bkg_new.arrays(library='pd')

    tree_sig_new = file_sig_new['ClusterTree']
    df_sig_new = tree_sig_new.arrays(library='pd')

    # Keep only bkg clusters in high pile-up events
    df_bkg_old = df_bkg_old[(df_bkg_old['avgMu'] > 20)] 
    df_bkg_new = df_bkg_new[(df_bkg_new['avgMu'] > 20)] 

    # Only keep the columns of interest
    columns = ['cluster_nCells_tot', 'clusterE','cluster_time', 
               'cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R', 
               'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_ENG_FRAC_EM', 
               'cluster_ISOLATION', 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 
               'cluster_SECOND_TIME'] 
    
    df_bkg_old = df_bkg_old[columns]
    df_sig_old = df_sig_old[columns]
    df_bkg_new = df_bkg_new[columns]
    df_sig_new = df_sig_new[columns]

    # Apply cuts
    apply_cuts(df_bkg_old)
    apply_cuts(df_sig_old)
    apply_cuts(df_bkg_new)
    apply_cuts(df_sig_new)

    ####################
    #  Plot features  #
    ###################

    print('Plotting features ... \n')
    with PdfPages(fig_path + '/features_comparison.pdf') as pdf:

        # Loop over field names
        for key in df_bkg_old:
            fig, ax = plt.subplots(1,2, figsize=[15., 5.])
            bins = 30
            ax1 = ax[0]
            ax2 = ax[1]
            _ , bin_edges, _ = ax1.hist(df_bkg_old[key].to_numpy(), bins=bins, histtype="step", 
                                      color = 'blue', density=True, label='Bkg, old data')

            _               = ax1.hist(df_bkg_new[key].to_numpy(), bins=bin_edges, histtype="step", 
                                      color = 'green', density=True, label='Bkg, new data')

            _                = ax2.hist(df_sig_old[key].to_numpy(), bins=bin_edges, histtype="step", 
                                      color = 'red',density=True, label='Sig, old data')
                
            _               = ax2.hist(df_sig_new[key].to_numpy(), bins=bin_edges, histtype="step", 
                                      color = 'orange', density=True, label='Sig, new data')

            # Features to put in log scale    
            if key in ['clusterE', 'cluster_CENTER_LAMBDA', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R',
                'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 'cluster_SECOND_TIME',  
                'cluster_nCells_tot']: 
                ax1.set_yscale('log')
                ax2.set_yscale('log')

            ax1.set_xlabel(key)
            ax2.set_xlabel(key)
            ax1.set_ylabel('Frequency')
            ax2.set_ylabel('Frequency')
            ax1.set_title('Pile up + hard scatter')
            ax2.set_title('Only hard scatter')
            ax1.legend()
            ax2.legend()

            # Save figure
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saving all figures into {fig_path}'/features_comparison.pdf' \n")

if __name__ == "__main__":
    main()