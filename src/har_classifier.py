import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 


import shutil
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

from models import *
import utils as utils

seed = 30
tf.random.set_seed(seed)
np.random.seed(seed)

"""
This script defines the main function for training the Har_Classifier model on a s
pecified dataset. It includes data loading, model instantiation, training with callbacks, 
and evaluation of metrics. The results are saved in a structured directory format, and logs are maintained for reproducibility.
"""
def clfOnlyTowers(directory, subdirectory, plots_loss_dir, data, obs, table, 
                 time_, dataset, n_epochs, scores, n_batch, cms, LABELS, ftune = False, 
                 with_val=False, modelname = '', modeltype = '', saveModel = True):
    
    print()
    print("Dataset: ", dataset)

    X_train = data['x_train']
    X_val = data['x_val']
    X_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    print("\nTraining the model from scratch...")

    if modeltype == 'har_classifier':
        # Instantiate the model
        classifier_model = Har_Classifier(num_classes=13)

        # Split data into accelerometer and gyroscope
        X_train_acc = X_train[:, :, :3]  # First 3 channels for accelerometer
        X_train_gyro = X_train[:, :, 3:6]  # Last 3 channels for gyroscope
        X_val_acc = X_val[:, :, :3]
        X_val_gyro = X_val[:, :, 3:6]
        X_test_acc = X_test[:, :, :3]
        X_test_gyro = X_test[:, :, 3:6]

        # Compile the model
        classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, weight_decay=1e-4), 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
                                     tf.keras.metrics.SparseCategoricalCrossentropy(name="sparse_categorical_crossentropy")])    
        
    best_model = "best_model_"+dataset+"_"+time_+".keras"
    path_best_model = subdirectory+"/"+best_model
    print("path_best_model: ", path_best_model)

    class CustomModelCheckPoint(tf.keras.callbacks.Callback):
        def __init__(self,**kargs):
            super(CustomModelCheckPoint,self).__init__(**kargs)
            self.epoch_accuracy = {} # loss at given epoch
            self.epoch_loss = {} # accuracy at given epoch
            self.epoch_validation = {}
            self.epoch_lossval = {}
            self.epoch_balanced_accuracy = {}  # balanced accuracy at given epoch

        def on_epoch_begin(self,epoch, logs={}):
            # Things done on beginning of epoch. 
            return

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_accuracy[epoch] = logs.get("accuracy")
            self.epoch_loss[epoch] = logs.get("loss")
            self.epoch_validation[epoch] = logs.get("val_accuracy")
            self.epoch_lossval[epoch] = logs.get("val_loss")
            self.epoch_balanced_accuracy[epoch] = logs.get("val_balanced_accuracy")

            if(dataset.startswith('de_fake_padts_100-9')):
                freq = 20
            else:
                freq = 5

            if ((epoch % freq)==0):
                utils.plot_loss_acc(epoch, self.epoch_accuracy, 
                              self.epoch_loss, self.epoch_validation, 
                              self.epoch_lossval, directory, dataset, 
                              time_, plots_loss_dir) #a random function
            

    chkpoint = CustomModelCheckPoint()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max')
    Reducelr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=15, verbose=1, min_delta=1e-4, mode='min')
    # earlyStopping = CustomEarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max', restore_best_weights=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=subdirectory+'/logs', histogram_freq=1, write_graph=True, write_images=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            path_best_model , save_freq=100, monitor='val_balanced_accuracy', mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            path_best_model , save_best_only=True, monitor='val_balanced_accuracy', mode='max'
        ),
        chkpoint,
        tensorboard_callback,
    ]

    # Training for classifier
    if modeltype == 'har_classifier':
        history = classifier_model.fit([X_train_acc, X_train_gyro], y_train,
                                  validation_data=([X_val_acc, X_val_gyro], y_val),
                                  batch_size=n_batch,
                                  epochs=n_epochs,
                                  callbacks=callbacks,
                                  verbose=1)

        # Save the trained model
        classifier_model.save(subdirectory+"/classifier_model.h5")

        # Read MSE per epoch
        train_mse = history.history['loss']
        val_mse   = history.history['val_loss']
        
        # Save the list of floats to the file
        file_path = os.path.join(subdirectory, "val_mse.txt")
        with open(file_path, "w") as f:
            for value in val_mse:
                f.write(f"{value}\n")

        for i, (tr, va) in enumerate(zip(train_mse, val_mse), 1):
            print(f"Epoch {i}: train MSE={tr:.6f}, val MSE={va:.6f}")

        # Plot
        plt.plot(train_mse, label='Train MSE')
        plt.plot(val_mse, label='Validation MSE')

        print("Test:")
        predictions_test = classifier_model.predict([X_test_acc, X_test_gyro])

        # Compute classification metrics
        metric_results_test = []
        utils.compute_all_metrics(y_test, predictions_test, dataset, time_, subdirectory, metric_results_test, 'test')

        # For validation
        predictions_val = classifier_model.predict([X_val_acc, X_val_gyro])
        metric_results_val = []
        utils.compute_all_metrics(y_val, predictions_val, dataset, time_, subdirectory, metric_results_val, 'val')

        # For train
        predictions_train = classifier_model.predict([X_train_acc, X_train_gyro])
        metric_results_train = []
        utils.compute_all_metrics(y_train, predictions_train, dataset, time_, subdirectory, metric_results_train, 'train')

    print(directory)
    print(subdirectory)
    print("Model name: ", modelname)


"""
Main function to run the training and evaluation of the Har_Classifier model. It sets up directories for saving models and logs, 
loads the dataset, processes it, and calls the training function. The results are saved in a structured format for later analysis.
"""
def main(modeltype, test_type, k_folds, norm_method, n_epochs, dataset, learning_rate, dropout_rate, 
         overlap_shift, n_batch, sensors, seg5, overlap, normalize, with_val):
    # Directory
    time_ = time.strftime("%Y%m%d-%H%M%S")
    directory = os.getcwd() + '/saved_models/' + 'recurrent_models_'+ time_

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Copy the submit script to the directory
    #shutil.copy('submit_job_model_patchi_classify.sh', directory)
    
    # Logging setup: Redirect stdout to console and file
    import sys
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    # Save log in the created directory
    log_path = os.path.join(directory, 'output.log')
    log_file = open(log_path, 'a')
    sys.stdout = Tee(sys.stdout, log_file)
    
    # Save script
    source_file = './'
    source_file = source_file + 'har_classifier.py'
    destination_folder = directory+"/har_classifier.py"
    print("Saving the code at: ", destination_folder)
    shutil.copy(source_file, destination_folder)


    dataset = dataset

    _, _, _, file_full_raws = utils.get_raw_datasets(dataset, magni=False)
    print(file_full_raws)

    if ( dataset == 'vivabem12_lying' ):
        df = utils.read_full_raws_without_tv(file_full_raws)
    elif ( dataset == 'vivabem12_tv' ):
        df = utils.read_full_raws_without_lying(file_full_raws)
    else:
        df = utils.read_full_raws(file_full_raws, dataset)

    obs = ''

    scores = []
    cms = []
    
    # Get the class names from dataframe
    LABELS = utils.get_class_names(dataset)
    fold = 1
    time_ = time.strftime("%Y%m%d-%H%M%S") 
    subdirectory = directory + '/' + dataset + '_'+str(fold)+'f_'+modeltype + '_'+time_
    print("subdirectory: ", subdirectory)
    print("directory: ", directory)
    if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

    # Define file path
    file_path = os.path.join(subdirectory, "hyperparams.txt")

    # Save string to file
    with open(file_path, "w") as f:
        f.write(f"modeltype: {modeltype}, test_type: {test_type}, k_folds: {k_folds}, norm_method: {norm_method}, dataset: {dataset}, learning_rate: {learning_rate}, dropout_rate: {dropout_rate}, overlap_shift: {overlap_shift}, n_batch: {n_batch}, n_epochs: {n_epochs}, sensors: {sensors}, seg5: {seg5}")
    
    print("Model type:", modeltype)

    plots_loss_dir = os.path.join(subdirectory,'plots_loss_dir')
    if not os.path.exists(plots_loss_dir):
        os.makedirs(plots_loss_dir)

    data = []
    print("With validation data")
    dict_arrays = utils.get_processed_fold(df, dataset, modeltype, subdirectory, sensors, fold,
                                            seg5, normalize, overlap, overlap_shift=overlap_shift)
    data = dict_arrays
        
    table = None  # Not used
    clfOnlyTowers(directory, subdirectory, plots_loss_dir, data, obs, table, 
                time_, dataset, n_epochs, scores,
                n_batch, cms, LABELS, ftune=False, with_val= with_val, 
                modelname=modeltype, modeltype=modeltype, saveModel=True )
    
    print("Model name: ", modeltype)
    print("Directory: ", directory)
    time_ = time.strftime("%Y%m%d-%H%M%S") 
    print("End of the program at: ", time_)

    # Restore original stdout
    sys.stdout = sys.stdout.files[0]
    # Close the log file
    log_file.close()
