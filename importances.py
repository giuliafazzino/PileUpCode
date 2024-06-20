import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pandas as pd

def get_feature_importance(test_X, test_y, model, n, recreate=False, dir_path='out'):
    """
    Evaluate the permutaion feature imortance for the training features of the model by shuffling one
    of them at a time and evaluating how much the chosen metric (AUC) is affected by it
    """
    
    feature_importance = []
    feature_importance_unc = []
    
    outfile = Path(dir_path + '/FI.npy')
    if outfile.is_file() and not recreate:
        feature_importance = np.load(dir_path = '/FI.npy')
        feature_importance_unc = np.load(dir_path = '/FI_unc.npy')

    else:

        # Generate default prediction
        print('Getting default prediction...')
        y_pred = model.predict(test_X, verbose=0)
        #labels = y_pred >= 0.6

        # Evaluate default performance for non permuated test data
        score_default = roc_auc_score(test_y, y_pred)

        # Loop over the different features
        for j in range(test_X.shape[1]):
            print('Feature ',j,' running...')

            score_with_perm = []

            # Iterate over the same feature several times
            for i in range(n):
                print('\t Iteration: ',i)

                # Generate a random permutation
                perm = np.random.permutation(range(test_X.shape[0]))

                # Generate a copy for which the one feature is permutated
                X_test_ = test_X.copy()
                X_test_[:, j] = test_X[perm, j]

                # Predict the labels with the permutated feature
                y_pred_ = model.predict(X_test_, verbose=0)

                # Evaluate performance and append to score_list
                s_ij = roc_auc_score(test_y, y_pred_)
                score_with_perm.append(s_ij)

            print('\n')

            # Save values 
            feature_importance.append(np.absolute(score_default - np.mean(score_with_perm)))
            feature_importance_unc.append(np.std(score_with_perm)) # Use std devation as uncertainity

        # Save arrays
        np.save(dir_path + '/FI.npy', feature_importance)
        np.save(dir_path + '/FI_unc.npy', feature_importance_unc)

    return feature_importance, feature_importance_unc


def plot_feature_importance(X, y, model, feature_names, iteration=10, recreate=False, dir_path='out'):
    """Calculate the features permutation importance over a given nnumber of repetitions 
    and then plot it in descending order"""

    plot_path = dir_path + "/plots"

    # Calculate feature importance
    feature_importance, uncertainty = get_feature_importance(X, y, model, iteration, recreate, dir_path)
    idx = np.argsort(feature_importance)
    uncertainty_sorted = np.array(uncertainty)[idx]

    # Plot importance
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.barh(range(X.shape[1]), np.sort(feature_importance), color="r", alpha=0.7)
    ax.set_yticks(range(X.shape[1]), np.array(feature_names)[idx])
    ax.errorbar(np.sort(feature_importance), range(X.shape[1]), xerr=uncertainty_sorted, fmt='o', color='black', alpha=0.7)
    ax.set_xlabel('Feature Importance')
    plt.title("Iterations, N="+str(iteration))

    # Save output
    plt.savefig(plot_path + '/feature_importance_'+str(iteration)+'.pdf')
    print('File ' + plot_path + '/feature_importance_'+str(iteration)+'.pdf has been created')


def main():
    dir_path = 'out'
    feature_names = ['cluster_nCells_tot', 'clusterE', 'cluster_time', 
               'cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R', 
               'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_ENG_FRAC_EM', 
               'cluster_ISOLATION', 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 
               'cluster_SECOND_TIME']     

    # Load model and data
    model = tf.keras.models.load_model(dir_path+"/model.h5", compile=False)
    data_test = pd.read_csv('data/df_test.csv')

    X = data_test[feature_names].to_numpy()
    y = data_test['label'].to_numpy()

    print('Checking data sizes...')
    print('\t X shape: ', X.shape)
    print('\t y shape: ', y.shape)

    # Plot importances and save
    plot_feature_importance(X, y, model, feature_names, 2, True, dir_path)

if __name__ == '__main__':
    main()
    pass