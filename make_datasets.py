"""
Read the .root files and create dataframes for training and testing
"""

import uproot as ur
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Matplotlib settings
params = {
    'font.size': 13
}
plt.rcParams.update(params)


def apply_cuts(df):
    """ Apply some cuts on the variables based on their physical meaning"""
    df = df.loc[df["cluster_ENG_CALIB_TOT"]>0.3]
    df = df.loc[df["clusterE"]>0.]
    df = df.loc[df["cluster_CENTER_LAMBDA"]>0.]
    df = df.loc[df["cluster_FIRST_ENG_DENS"]>0.]
    df = df.loc[df["cluster_SECOND_TIME"]>0.]
    df = df.loc[df["cluster_SIGNIFICANCE"]>0.]
    return df


def plot_features(df, output_path):
    """ Plot all the features in the dataframe, both in linar and log scale, and save output"""
    with PdfPages(output_path) as pdf:

        # Loop over field names
        for idx, key in enumerate(df):
            print(f'Accessing variable with name = {key} ({idx+1} / {df.shape[1]})')
            data = df[key].to_numpy()
            if key != 'label':
            # Make plots 
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[5.5*3, 5.])

            # Linear scale
                bins = 30
                _, bin_edges, _ = ax1.hist(data, bins=bins, histtype="step", density=False, label="linear")
                ax1.set_xlabel(key)
                ax1.set_ylabel("Frequency")

            # Log scale for y-axis
                if data.min() <= 0:
                    data_upshifted = data + np.abs(data.min()) + 1e-30
                    label = "log (upshifted)"
                else:
                    data_upshifted = data
                    label = "log"

                ax2.hist(data_upshifted, bins=bins, histtype="step", density=False, label=label)
                ax2.set_yscale("log")
                ax2.set_xlabel(key)
                ax2.legend(frameon=False, loc="upper right")
                ax2.set_ylabel("Frequency")

                # Log scale for both axes
                bins_log = np.logspace(np.log10(data_upshifted.min()), np.log10(data_upshifted.max()), len(bin_edges)-1)
                ax3.hist(data_upshifted, bins=bins_log, histtype="step", density=False, label=label)
                ax3.set_yscale("log")
                ax3.set_xscale("log")
                ax3.set_xlabel(key)
                ax3.legend(frameon=False, loc="upper right")
                ax3.set_ylabel("Frequency")
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saving all figures into {output_path} \n")


def plot_features_classes(dataframes, labels, output_path):
    """ Plot the features for both dataframes (bkg and signal), and save output"""

    with PdfPages(output_path) as pdf:

        # Loop over field names
        for key in dataframes[0]:

            if key != 'label':
                fig, ax = plt.subplots(figsize=[10., 5.])
                df_bkg = dataframes[0]
                df_sig = dataframes[1]
                bins = 30

                _, bin_edges, _ = ax.hist(df_bkg[key].to_numpy(), bins=bins, 
                                        histtype="step", color = 'blue', density=True, label=labels[0])
                
                _               = ax.hist(df_sig[key].to_numpy(), bins=bin_edges, 
                                        histtype="step", color = 'red', density=True, label=labels[1])
                
                if key in ['clusterE', 'cluster_CENTER_LAMBDA', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R',
                   'cluster_AVG_LAR_Q',
                   'cluster_AVG_TILE_Q', 'cluster_SECOND_TIME',  
                    'cluster_nCells_tot', 'r_e_calculated']: #'cluster_fracE',
                    ax.set_yscale('log')

                ax.set_xlabel(key)
                ax.set_ylabel('Frequency')
                ax.legend()


                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saving all figures into {output_path} \n")


def normalize(x, pre_derived_scale = None):
    if pre_derived_scale:
        mean, std = pre_derived_scale[-2], pre_derived_scale[-1]
    else:
        mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    return out, mean, std


def apply_log(x, pre_derived_scale = None):

    # Shift up if x-min < 0
    epsilon = 1e-10
    minimum = x.min()

    if minimum <= 0:
        x = x - minimum + epsilon
    else:
        minimum = 0
        epsilon = 0

    return np.log(x), minimum, epsilon


def apply_scale(df, field_name, mode, pre_derived_scale = None):
    """ Re-scale the variables"""
    if pre_derived_scale:
        old_scale = pre_derived_scale
    else:
        old_scale = None

    if mode=='lognormalise':
        x, minimum, epsilon = apply_log(df[field_name], old_scale)
        x, mean, std = normalize(x)
        scale = ("SaveLog / Normalize", minimum, epsilon, mean, std)

    elif mode=='normalise':
        x = df[field_name]
        x, mean, std = normalize(x, old_scale)
        scale = ("Normalize", mean, std)

    elif mode=='special':
        x = df[field_name]
        x = np.abs(x)**(1./3.) * np.sign(x)
        x, mean, std = normalize(x, old_scale)
        scale = ("Sqrt3", mean, std)

    else:
        raise ValueError('Scaling mode need for ', field_name)
    
    df[field_name] = x
    return df, scale

def calculate_response(df):
    resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
    df = df.assign(r_e_calculated = resp)
    df = df.loc[df['r_e_calculated']>0.1]
    return df


def main():
    # Import files
    #file_bkg = ur.open('/eos/user/g/gfazzino/pileupdata/fromChris/moreStats_mltree_mc20e_withPileup.root') 
    #file_sig = ur.open('/eos/user/g/gfazzino/pileupdata/fromChris/moreStats_mltree_mc20e_noPileup.root') 
    file_bkg = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc20e/mc20e_withPU.root') 
    file_sig = ur.open('/eos/user/g/gfazzino/pileupdata/SamplesForGiulia/mc20e/mc20e_noPU.root') 
    print('Found files, reading datasets... \n')

    # Output paths
    out_path = 'out'
    fig_path = out_path +'/plots'
    data_path = 'data'

    # Output folders
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

    try:
        os.system("mkdir {}".format(data_path))
    except ImportError:
        print("{} already exists \n".format(data_path))
    pass



    ###########################
    #  Dataframe Preparation  #
    ###########################

    # Merge data and add labels
    tree_bkg = file_bkg['ClusterTree']
    df_bkg = tree_bkg.arrays(library='pd')
    df_bkg.insert(0, 'label', [0 for i in range(df_bkg.shape[0])], True)
    
    tree_sig = file_sig['ClusterTree']
    df_sig = tree_sig.arrays(library='pd')
    df_sig.insert(0, 'label', [1 for i in range(df_sig.shape[0])], True)

    # Plot bkg number of vertexes
    fig, ax = plt.subplots(figsize=[10., 5.])
    _, bin_edges, _ = ax.hist(df_bkg['avgMu'].to_numpy(), bins = 45, 
                              histtype="step", color = 'blue')

    ax.set_xlabel('<$\mu$>')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 250000)

    plt.savefig(fig_path + '/avg_mu.pdf')
    plt.close(fig)
    print('File ' + fig_path + '/avg_mu.pdf has been created \n')

    # Keep only clusters in high pile-up events
    df_bkg = df_bkg[(df_bkg['avgMu'] > 20)] 

    # Merge
    df = pd.concat([df_bkg, df_sig], ignore_index = True)


    # Apply cuts
    df = apply_cuts(df)
    df_bkg = apply_cuts(df_bkg)
    df_sig = apply_cuts(df_sig)

    #pd.set_option('use_inf_as_na',True)

    df.dropna(inplace=True)

    # Calculate response 
    df = calculate_response(df)
    df_bkg = calculate_response(df_bkg)
    df_sig = calculate_response(df_sig)


    # Remove problematic values
    df = df.loc[df['r_e_calculated'] < 1e10]
    df_bkg = df_bkg.loc[df_bkg['r_e_calculated'] < 1e10]
    df_sig = df_sig.loc[df_sig['r_e_calculated'] < 1e10]
    

    # Only keep the columns we want
    columns = ['label','cluster_nCells_tot', 'clusterE','cluster_time', 
               'cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R', 
               'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_ENG_FRAC_EM', 
               'cluster_ISOLATION', 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 
               'cluster_SECOND_TIME', 
                # these ones are not used for PU suppression
               'clusterEta', 'cluster_SIGNIFICANCE', 'cluster_PTD', 'nPrimVtx', 
               'cluster_ENG_CALIB_TOT', 'r_e_calculated', 'cluster_LONGITUDINAL' , 'avgMu']      
    
    df = df[columns]
    df_bkg = df_bkg[columns]
    df_sig = df_sig[columns]

    # Shuffle the rows
    df = df.sample(frac=1).reset_index(drop=True)


    bkg = df['label'] == 0
    print(f'Total number of clusters = {df.shape[0]}')
    print(f'Number of clusters with pile-up = {df[bkg].shape[0]}')
    print(f'Number of no pile-up clusters = {df[~bkg].shape[0]} \n')

    # Split data for training and testing
    n_train = int(0.8*df.shape[0])
    df_train = df[:n_train]
    df_test = df[n_train:]

    print(f'Number of training clusters = {df_train.shape[0]}')
    print(f'Number of testing clusters = {df_test.shape[0]} \n')

    ###############################################
    #  Save number of clusters infos on txt file  #
    ###############################################
    info_file = open(out_path + '/infos.txt', 'w') 

    info_file.write(f'Total number of clusters = {df.shape[0]}\n')
    info_file.write('-'*100 + '\n')
    info_file.write(f'Number of clusters with pile-up = {df[bkg].shape[0]}\n')
    info_file.write(f'Number of no pile-up clusters = {df[~bkg].shape[0]}\n')

    info_file.write('-'*100 + '\n')
    bkg_train = df_train['label'] == 0
    info_file.write(f'Number of training clusters = {df_train.shape[0]}\n')
    info_file.write(f'With pile-up = {df_train[bkg_train].shape[0]}\n')
    info_file.write(f'Without pile-up = {df_train[~bkg_train].shape[0]}\n')

    info_file.write('-'*100 + '\n')
    bkg_test = df_test['label'] == 0
    info_file.write(f'Number of test clusters = {df_test.shape[0]}\n')
    info_file.write(f'With pile-up = {df_test[bkg_test].shape[0]}\n')
    info_file.write(f'Without pile-up = {df_test[~bkg_test].shape[0]}\n')

    info_file.close()
    print('File ' + out_path + '/infos.txt has been created \n')

    print('Reading datasets completed \n')

    ######################
    #  Look at features  #
    ######################

    print('Features: \n', df.columns, '\n')

    # Plot features as they are
    print('Plotting the features ... \n')
    plot_features(df, fig_path+'/features.pdf')
    plot_features_classes([df_bkg, df_sig], ['Hard scatter & pile-up','No pile-up'], fig_path+'/features_classes.pdf')
    # Plot features for in time vs out of time clusters
    oot_mask = np.abs(df_bkg['cluster_time']) > 12.5
    plot_features_classes([df_bkg[oot_mask], df_bkg[~oot_mask]], ['t > 12.5 ns', r't $\leq $ 12.5 ns'], fig_path+'/features_timing.pdf')

    # Plot correlation matrix
    print('Plotting the correlation matrix ...')
    fig, ax = plt.subplots(figsize=(20,22))
    sns.heatmap(df.drop('label', axis = 1).corr(), annot = True, fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 'horizontal')
    plt.savefig(fig_path + '/correlation.pdf')
    print('Saving figure in ' + fig_path+'/correlation.pdf \n')

    ################### 
    #  Preprocessing  #
    ###################
    
    file_scales = data_path + '/all_info_df_scales.txt'

    print("Making preprocessing ... \n")
    scales = {}

    # Log-preprocessing
    field_names = ['clusterE', 'cluster_CENTER_LAMBDA', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R',
                   'cluster_AVG_LAR_Q',
                   'cluster_AVG_TILE_Q', 'cluster_SECOND_TIME', 
                   'cluster_nCells_tot', 'cluster_SIGNIFICANCE'] #'cluster_fracE', 
    for field_name in field_names:
        df_train, scales[field_name] = apply_scale(df_train, field_name, 'lognormalise')
        df_test = apply_scale(df_test, field_name, 'lognormalise', scales[field_name])[0]

    # Just normalizing
    field_names = ['cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM',
                   'cluster_LATERAL', 'cluster_ISOLATION', 'clusterEta', 'nPrimVtx', 'cluster_PTD', 
                   'avgMu', 'cluster_LONGITUDINAL']
    for field_name in field_names:
        df_train, scales[field_name] = apply_scale(df_train, field_name, 'normalise')
        df_test = apply_scale(df_test, field_name, 'normalise', scales[field_name])[0]

    # Special preprocessing
    field_name = "cluster_time"
    df_train, scales[field_name] = apply_scale(df_train, field_name, 'special')
    df_test = apply_scale(df_test, field_name, 'special', scales[field_name])[0]

    # Make plots after preprocessing
    plot_features(df_train, fig_path+'/features_scaled.pdf')
    

    ############
    #  Saving  #
    ############
    save = True

    if save:
        #arr_train = df_train.to_numpy()
        #arr_test = df_test.to_numpy()

        #print('Training array shape: ', arr_train.shape)
        #print('Testing array shape: ', arr_test.shape, '\n')

        # Save scales
        with open(file_scales, "w") as f:
            f.write(str(scales))

        # Save data    
        print('Saving training data ... \n')
        df_train.to_csv('data/df_train.csv', index = False)
        print('Saving test data ...')
        df_test.to_csv('data/df_test.csv', index = False)
    




if __name__ == "__main__":
    main()