import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


# Matplotlib settings
params = {
    'font.size': 13
}
plt.rcParams.update(params)

def main():
    # Read the datasets
    dir_path = 'out'
    output_path = dir_path + '/plots'

    #data_path = 'data'
    #data_train = pd.read_csv(data_path + '/df_train.csv') 
    #data_test = pd.read_csv(data_path + '/df_test.csv') 

    x_test = np.load(dir_path + '/x_test.npy')
    y_test = np.load(dir_path + '/trueClass_test.npy')
    p_test = np.load(dir_path + '/predictions_test.npy')

    x_train = np.load(dir_path + '/x_train.npy')
    y_train = np.load(dir_path + '/trueClass_train.npy')
    p_train = np.load(dir_path + '/predictions_train.npy')

    # Keep mis-id signal clusters below the cut
    cut = 0.4
    mask_test = (y_test == 1) & (p_test < cut)
    mask_train = (y_train == 1) & (p_train < cut)
    x_misid = np.concatenate((x_test[mask_test], x_train[mask_train]))

    # Feature names
    feature_names = ['cluster_nCells_tot', 'clusterE', 'cluster_time', 
               'cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R', 
               'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_ENG_FRAC_EM', 
               'cluster_ISOLATION', 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 
               'cluster_SECOND_TIME']      
    
    
    # Plot the features for just the misid clusters
    with PdfPages(output_path+'/features_misid.pdf') as pdf:

        # Loop over field names
        for idx, key in enumerate(feature_names):
            print(f'Accessing variable with name = {key} ({idx+1} / {x_misid.shape[1]})')
            data = x_misid[idx,:]
            # Make plots 
            fig, ax1 = plt.subplots(1, 1, figsize=[10., 5.])

            # Linear scale
            bins = 10
            _, bin_edges, _ = ax1.hist(data, bins=bins, histtype="step", density=False)
            ax1.set_xlabel(key)
            ax1.set_ylabel("Frequency") 
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    # Compare mis-id clusters features to pile-up features
    with PdfPages(output_path+'/features_misid_comp.pdf') as pdf:

        # Loop over field names
        for idx, key in enumerate(feature_names):
            print(f'Accessing variable with name = {key} ({idx+1} / {x_misid.shape[1]})')
            data = x_misid[idx,:]
            data_PU = x_test[y_test == 0][idx,:]
            data_noPU = x_test[y_test == 1][idx,:]

            # Make plots 
            fig, ax1 = plt.subplots(1, 1, figsize=[10, 5.])

            # Linear scale
            bins = 20
            _, bin_edges, _ = ax1.hist(data, bins=bins, histtype="stepfilled", alpha = .5, density=True, label="mis-id hard scatter clusters")
            _ = ax1.hist(data_PU, bins=bin_edges, histtype="stepfilled", alpha = .5, density=True, label="clusters with pile-up")
            _ = ax1.hist(data_noPU, bins=bin_edges, histtype="stepfilled", alpha = .5, density=True, label="hard scatter clusters")
            ax1.legend()
            ax1.set_xlabel(key)
            ax1.set_ylabel("Frequency") 
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saving all figures into {output_path}/features_misid_comp.pdf \n")
    
    return


if __name__ == "__main__":
    main()