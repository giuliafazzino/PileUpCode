"""
Train and test the DNN, depending on the first argument (--train or --test), and save the otput  in the directory specified by--outdir
"""
##############
#  Includes  #
##############
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight


# Matplotlib settings  
params = {
    'font.size': 14
}
plt.rcParams.update(params)

####################
#  Parser options  #
####################
parser = argparse.ArgumentParser()
parser.add_argument('--train',    dest='train',    action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--test',     dest='test',     action='store_const', const=True, default=False, help='Test NN   (default: False)')
parser.add_argument('--outdir',   dest='outdir',   type=str, default='out', help='Directory with output is stored')

args = parser.parse_args()

##########
#  Main  #
##########
def main():

    # Output directory
    out_path = args.outdir

    ###############
    #  Read data  #
    ###############
    classification_features = ['cluster_nCells_tot', 'clusterE','cluster_time', 
               'cluster_EM_PROBABILITY', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_R', 
               'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_ENG_FRAC_EM', 
               'cluster_ISOLATION', 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q', 
               'cluster_SECOND_TIME']

    # Train (label is in the first column)
    print('-'*100)
    data_train = pd.read_csv('data/df_train.csv')
    x_train = data_train[classification_features].to_numpy()
    y_train = data_train['label'].to_numpy()
    print(f'Training dataset size {y_train.shape}')
    print(f'Signal events {np.sum(y_train)}')

    # Test (label is in the first column)
    data_test = pd.read_csv('data/df_test.csv')
    x_test = data_test[classification_features].to_numpy()
    y_test = data_test['label'].to_numpy()
    print(f'Test dataset size {y_test.shape}')
    print(f'Signal events {np.sum(y_test)}')
    print('-'*100, '\n')

    ###########
    #  Train  #
    ###########
    if args.train:

        ###############
        #  Build DNN  #
        ###############
        nepochs = 4
        
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30, start_from_epoch = 100, restore_best_weights=False)
        
        model = keras.Sequential([keras.layers.Flatten(input_shape=(x_train.shape[1],)),
                                             keras.layers.Dense(512, activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             #keras.layers.Dropout(.3),

                                             keras.layers.Dense(256, activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             #keras.layers.Dropout(.3),

                                             keras.layers.Dense(128, activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             keras.layers.Dropout(.3),

                                             keras.layers.Dense(64, activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             keras.layers.Dropout(.3),

                                             keras.layers.Dense(32,  activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             #keras.layers.Dropout(.3),
            

                                             keras.layers.Dense(16,   activation="relu"),
                                             keras.layers.BatchNormalization(),
                                             #keras.layers.Dropout(.3),
                                             #keras.layers.Dense(4,   activation="relu"),

                                             keras.layers.Dense(1,   activation="sigmoid")])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001), loss='binary_crossentropy', 
                      metrics=[keras.metrics.AUC()])
        
        # Reweighting
        weights = class_weight.compute_class_weight(class_weight = 'balanced',  classes = np.unique(y_train), 
                                                    y = y_train)
        
        class_weights = {0: weights[0], 
                         1: weights[1]}
        
        
        # Fit model to data
        history = model.fit(x_train, y_train, validation_split = 0.25, epochs=nepochs, batch_size = 1024, 
                            class_weight = class_weights, 
                            callbacks = [early_stop])
        
        metrics = pd.DataFrame({"Train_Loss":history.history['loss'],"Val_Loss":history.history['val_loss']})
        metrics.to_csv(out_path + '/Losses_train.csv', index = False)

        ##########
        #  Loss  #
        ##########
        fig = plt.figure(figsize = [12,6])
        gs = fig.add_gridspec(2, hspace = .1, height_ratios = [1, .3])
        ax = gs.subplots(sharex=True, sharey=False)
        ax[0].plot(history.history['loss'], 'bo', label='loss', markersize=1.5, linestyle='dashed',)
        ax[0].plot(history.history['val_loss'], 'go', label='val_loss', markersize=1.5, linestyle='dashed',)

        ax[1].plot(np.array(history.history['val_loss']) - np.array(history.history['loss']), 'bo', 
                   markersize = 2, linestyle='dashed', label = 'val_loss - train_loss')
        
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylim(-.03, .03)

        # Show only ticks and labels in the outer sides of the plots
        for a in ax:
            a.label_outer()

        ax[0].legend()
        ax[1].legend()
        ax[0].grid(True)
        ax[1].grid(True)

        plt.savefig(out_path + '/plots/Losses_train.png')
        plt.clf()

        ############
        #  Saving  #
        ############
        print('Saving output ...')

        model.save(out_path+"/model.h5")
        data_train = data_train.assign(pred = model.predict(data_train[classification_features].to_numpy()).flatten())
        data_train.to_csv('data/df_train.csv', index = False)

    ##############
    #    Test    #
    ##############  
    if args.test:

        # Load model
        model = keras.models.load_model(out_path+"/model.h5", compile=False)
        if model == None:
            return
            
        # Save outputs
        print('Calculating predisctions and saving output ...')
        data_test = data_test.assign(pred = model.predict(data_test[classification_features].to_numpy()).flatten())
        data_test.to_csv('data/df_test.csv', index = False)
                
    return


if __name__ == '__main__':
    main()
    pass
