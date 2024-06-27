import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import uproot as ur
import datetime, os
import tensorflow as tf
from sklearn.preprocessing import  QuantileTransformer, StandardScaler
from sklearn.utils import class_weight

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn import ensemble

from keras.backend import sigmoid
import uproot
from keras import regularizers
import keras.backend as K


from keras.utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def swish(x, beta = 1):
    return 2*(x * sigmoid(beta * x))

def tanhPlusOne(x):
    return 2*(tf.keras.activations.tanh(x) + 1)
    # return 6*(tf.keras.activations.tanh(x))

def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Below in place of swish you can take any custom key for the name
get_custom_objects().update({'swish': swish})
get_custom_objects().update({'tanhPlusOne': tanhPlusOne})

parser = argparse.ArgumentParser(description='Perform signal injection test.')
parser.add_argument('--train',    dest='train',    action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--retrain',  dest='retrain',  action='store_const', const=True, default=False, help='Retrain NN  (default: False)')
parser.add_argument('--test',     dest='test',     action='store_const', const=True, default=False, help='Test NN   (default: False)')
parser.add_argument('--closure',  dest='closure',  action='store_const', const=True, default=False, help='Closure')
parser.add_argument('--weight',   dest='weight',   type=int, default=0, help='Weight: 0: no weight 1: response, 2: response wider, 3: energy, 4: log energy')
parser.add_argument('--outdir',   dest='outdir',   type=str, default='', help='Directory with output is stored')

args = parser.parse_args()

def loss_wrapper(Ecluster): # NOT WORKING -- TO BE FIXED
    def lgk_loss_function_modified(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
        alpha = tf.constant(0.05)
        bandwith = tf.constant(0.1)
        pi = tf.constant(math.pi)
        ## LGK (h and alpha are hyperparameters)
        norm = -1/(bandwith*tf.math.sqrt(2*pi))
        gaussian_kernel  = norm * tf.math.exp( -(y_pred/y_true - 1)**2 / (2*(bandwith**2)))
        leakiness = alpha*tf.math.abs(y_pred/y_true - 1)
        lgk_loss = gaussian_kernel + leakiness
        loss = lgk_loss
        if Ecluster/y_pred > 0.3:
            loss += 1e10
        return loss
    return lgk_loss_function_modified

def lgk_loss_function(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
    alpha = tf.constant(0.05)
    bandwith = tf.constant(0.1)
    pi = tf.constant(math.pi)
    ## LGK (h and alpha are hyperparameters)
    norm = -1/(bandwith*tf.math.sqrt(2*pi))
    gaussian_kernel  = norm * tf.math.exp( -(y_pred/y_true - 1)**2 / (2*(bandwith**2)))
    leakiness = alpha*tf.math.abs(y_pred/y_true - 1)
    lgk_loss = gaussian_kernel + leakiness
    loss = lgk_loss
    return loss


def lgk_loss_function_1(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
    alpha = tf.constant(0.05)
    bandwith = tf.constant(0.1)
    pi = tf.constant(math.pi)
    ## LGK (h and alpha are hyperparameters)
    norm = -1/(bandwith*tf.math.sqrt(2*pi))
    gaussian_kernel  = norm * tf.math.exp(  -(y_pred/y_true - 1)**2 / (2*(bandwith**2)))
    lgk_loss = gaussian_kernel 
    loss = lgk_loss
    return loss


def build_and_compile_model(X_train, lr):
    model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
                                             layers.Dense(256, activation="swish"),
                                             layers.Dense(128, activation="swish"),
                                             layers.Dense(64,  activation="swish"),
                                             layers.Dense(8,   activation="swish"),
                                             layers.Dense(1,   activation="relu")]) #'tanhPlusOne')]) #
    model.compile(loss=lgk_loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    #model.compile(loss=loss_wrapper(X_train[:,1]), optimizer=tf.keras.optimizers.Adam(learning_rate=lr)) #, weighted_metrics=[]
    return model

# Main function.
def main():

    dir_path = args.outdir
    try:
        os.system("mkdir {}".format(dir_path))
    except ImportError:
        print("{} already exists".format(dir_path))
    pass

    #good_list_16  = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # the first 2 zeros should be ones
    good_list_11 = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  

    feature_names=['r_e_calculated',  'clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 
                   'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                    'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_time', 
                    'cluster_ISOLATION',  'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu']
    
    feature_names_11=['clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 
                   'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                    'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SIGNIFICANCE', 'nPrimVtx']

    # train dataset
    dataset_train = np.load('data/all_info_df_train.npy')
    x_train_11 = dataset_train.compress(good_list_11, axis=1)
    y_train = dataset_train[:, 0]
    print(f"Training dataset size {y_train.shape}")


    # test dataset
    print('Reading test datasets...\n')
    dataset_test = np.load('data/all_info_df_test.npy')
    x_test_11 = dataset_test.compress(good_list_11, axis=1)
    y_test = dataset_test[:, 0]


    print(f"Test dataset size {y_test.shape}")

    if args.train:
        nepochs = 100
        with tf.device('/GPU:0'):
            dnn_model = build_and_compile_model(x_train_11, lr=0.0001)
            history = dnn_model.fit( x_train_11, y_train, validation_split=0.35, epochs=nepochs, batch_size=4096) # batch_size=1024 

        metrics = pd.DataFrame({"Train_Loss":history.history['loss'],"Val_Loss":history.history['val_loss']})
        metrics.to_csv('out/Losses_train_redRelu.csv', index = False)

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_train_redRelu.png')
        plt.clf()
        dnn_model.save(dir_path+"/model_redRelu.h5")

    if args.retrain:
        nepochs = 100
        #with tf.device('0'):
        dnn_model = tf.keras.models.load_model(dir_path+"/model_redRelu.h5", compile=False, custom_objects={'swish': swish, 'tanhPlusOne': tanhPlusOne})
        dnn_model.compile(loss=lgk_loss_function_1, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), weighted_metrics=[])
        history = dnn_model.fit(x_train_11, y_train, validation_split=0.35, epochs=nepochs, batch_size=4096) # batch_size=1024 

        metrics = pd.DataFrame({"Train_Loss":history.history['loss'],"Val_Loss":history.history['val_loss']})
        metrics.to_csv('out/Losses_final_redRelu.csv', index = False)

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_final_redRelu.png')
        plt.clf()
        dnn_model.save(dir_path+"/model_redRelu.h5")

    ## testing
    if args.test:
        dnn_model = tf.keras.models.load_model(dir_path+"/model_redRelu.h5", compile=False)
        if dnn_model == None:
            return
        print('Calculating predictions...')
        #y_pred = dnn_model.predict(x_test_16).flatten()
        y_pred = dnn_model.predict(x_test_11).flatten()
                
        ### start saving output
        print('Saving outputs...')
        np.save(dir_path + '/trueResponse_redRelu.npy', y_test)
        np.save(dir_path + '/predResponse_redRelu.npy', y_pred)
        np.save(dir_path + '/x_test.npy', x_test_11)

    if args.closure:
        dnn_model = tf.keras.models.load_model(dir_path+"/model_redRelu.h5", compile=False)
        if dnn_model == None:
            return
        y_pred = dnn_model.predict(x_train_11).flatten()
        ### start saving output
        np.save(dir_path + '/trueResponse_redRelu_closure.npy', y_train)
        np.save(dir_path + '/predResponse_redRelu_closure.npy', y_pred)
        np.save(dir_path + '/x_train_redRelu.npy', x_train_11)

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
