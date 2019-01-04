#!/usr/bin/env python
# Script to train and test a neural network with TF's Keras API
from __future__ import division #idk if you need this
import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf
import saddle_function_utils as sfu
import math


def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """
    # TO-DO. Remove the two lines below and complete this function by computing the real mean and std. dev for the data
    # mean = np.zeros((data.shape[1],))
    # stdev = np.ones((data.shape[1],))

    mean_vars = [] #variables used to caluclate mean
    mean_list = [] #list of the means
    stdev_list = [] #list of the stdevs

    num_rows = data.shape[0]    #.shape[0] = # of rows
    num_columns = data.shape[1] #.shape[1] = # of columns
 
    for j in range(num_columns):
        for i in range(num_rows):
            mean_vars.append(data[i][j])  
            #adds each element to list mean_vars       
        mean_calc = np.mean(mean_vars) #calculates the mean of the vars in the list
        mean_list.append(mean_calc)  #adds this to the list for 



        stdev_calc = np.std(mean_vars) #calculates the stdev of the vars in the list
        stdev_list.append(stdev_calc) # adds this to the list for stdev
        mean_vars = [] #now need to wipe the list before we do the calculation again
            #note to self about the stdev calculation
            #so I think this is working, but according to an online formula, it's calculating 
            #the "population standard deviation" and not the "sample standard deviation", 
            #idk the difference and if it matters right now but we'll see with the results later

    mean = np.array(mean_list)
    stdev = np.array(stdev_list)
    
    mean = mean.reshape((1,mean.size))
    stdev = stdev.reshape((1,stdev.size))

    return mean, stdev


def normalize_data_per_row(data, mean, stdev):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :param mean: mean for normalization
    :param stdev: standard deviation for normalization
    :return: whitened data, (data - mean) / stdev
    """
    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    # TODO. Complete. Replace the line below with code to whitten the data.
    normalized_data = data

    for j in range(normalized_data.shape[1]): #column
        for i in range(normalized_data.shape[0]): #row
            #[i][j] = [ROW][COLUMN]
            normalized_data[i][j] = (normalized_data[i][j] - mean[0][j])/stdev[0][j]  
            #normalized_data = (data - mean) / StDev

    return normalized_data


def build_linear_model(num_inputs):
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    # TO-DO: Complete. Remove the None line below, define your model, and return it.

    input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")
    hidden1 = tf.keras.layers.Dense(64, use_bias=True)(input)
    output = tf.keras.layers.Dense(1, use_bias=True)(hidden1)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")

    model.summary()
    return model

def build_nonlinear_model(num_inputs):
    """
    Build nonlinear NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")
    #hidden1 = tf.keras.layers.Dense(64, use_bias=True)(input)
    hidden1 = tf.keras.layers.Dense(64, activation='relu')(input)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(64, activation='relu')(hidden2)
    output = tf.keras.layers.Dense(1, use_bias=True)(hidden3)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")

    model.summary()
    return model


def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """
    # TO-DO. Complete. Remove the pass line below, and add the necessary training code.

    # first normalize the input features in the training and validation set using 
    # the normalize_data_per_row() function from step 1 above.

    # normalize
    norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # THEN compile the neural network model that is passed as input to the function, 
    #i.e., define the optimizer to be used during training, loss, and relevant metrics. 

     # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='mse',
                 metrics=['mae'])

    # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_monkey_weights.h5'),
                                                            monitor='val_loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            period=1)


    # FINALLY, the train_model() function should train the network's weights using the 
    #model's fit function.
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(norm_val_input, val_target),
             callbacks=[tbCallBack, checkpointCallBack])

    #pass


def test_model(model, test_input, test_target, input_mean, input_stdev, batch_size=60):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param test_target: test targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # TODO. Complete. Remove the return line below and add the necessary code to make predictions with the model.

    #test_input = data
    #input_mean = mean
    #input_stdev = stdev

    to_predict = normalize_data_per_row(test_input, input_mean, input_stdev)
    prediction = model.predict(to_predict)
    #print("X=%s, Predicted=%s" % (to_predict[0], prediction[0]))
    # model.predict(
    #     x,
    #     batch_size=None,
    #     verbose=0,
    #     steps=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False
    # )
    #the goal/ what was given as place holder: np.zeros(test_target.shape)
    return prediction


def compute_average_L2_error(test_target, predicted_targets):
    """
    Compute the average L2 error for the predictions
    :param test_target: matrix with ground truth targets [N x 1]
    :param predicted_targets: matrix with predicted targets [N x 1]
    :return: average L2 error
    """
    # TO-DO. Complete. Replace the line below with code that actually computes the average L2 error over the targets.
    #average_l2_err = 0
    #.sqrt(x)
    #.pow(x,2) = x raised to 2 power

    diff = 0 #holds the difference variable
    l2_err = 0; #will hold the l2_err; is going to be square rooted at the end

    for j in range(test_target.shape[1]): #column
        for i in range(test_target.shape[0]): #row
            #[i][j] = [ROW][COLUMN]
            diff = test_target[i][j] - predicted_targets[i][j]
            l2_err = l2_err + math.pow(diff,2)
 
    average_l2_err = math.sqrt(l2_err)

    return average_l2_err


def main(num_examples, epochs, lr, visualize_training_data, build_fn=build_linear_model, batch_size=16):
    """
    Main function
    :param num_training: Number of examples to generate for the problem (including training, testing, and val)
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param visualize_training_data: visualize the training data?
    """

    np.random.seed(0) # make the generated values deterministic. do not change!
    
    # generate data
    monkey_function = lambda x: np.power(x[0], 3) - 3*x[0]*np.power(x[1],2)
    input, target = sfu.generate_data_for_2D_function(monkey_function, N=num_examples)

    # split data into training (70%) and testing (30%)
    all_train_input, all_train_target, test_input, test_target = sfu.split_data(input, target, 0.6)

    # visualize all training/testing (uncomment if you want to visualize the whole dataset)
    # plot_train_and_test(all_train_input, all_train_target, test_input, test_target, "train", "test", title="Train/Test Data")

    # split training data into actual training and validation
    train_input, train_target, val_input, val_target = sfu.split_data(all_train_input, all_train_target, 0.8)

    # visualize training/validation (uncomment if you want to visualize the training/validation data)
    if visualize_training_data:
        sfu.plot_train_and_test(train_input, train_target, val_input, val_target, "train", "validation", title="Train/Val Data")

    # normalize input data and save normalization parameters to file
    mean, stdev = compute_normalization_parameters(train_input)

    # build the model
    model = build_fn(train_input.shape[1])

    # train the model
    print "\n\nTRAINING..."
    train_model(model, train_input, train_target, val_input, val_target, mean, stdev,
                epochs=epochs, learning_rate=lr, batch_size=batch_size)

    # test the model
    print "\n\nTESTING..."
    predicted_targets = test_model(model, test_input, test_target, mean, stdev)

    # Report average L2 error
    l2_err = compute_average_L2_error(test_target, predicted_targets)
    print "L2 Error on Testing Set: {}".format(l2_err)

    # visualize the result (uncomment the line below to plot the predictions)
    sfu.plot_test_predictions(test_input, test_target, predicted_targets, title="Predictions")


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="total number of examples (including training, testing, and validation)",
                        type=int, default=600)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=16)
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=50)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=50)
    parser.add_argument("--visualize_training_data", help="visualize training data",
                        action="store_true")
    parser.add_argument("--build_fn", help="model to train (e.g., 'linear')",
                        type=str, default="linear")
    parser.add_argument("--load_model", help="path to the model",
                    type=str, default="")
    ##I added ^^^ this last parser line
    args = parser.parse_args()

    # define the model function that we will use to assemble the Neural Network
    if args.build_fn == "linear":
        build_fn = build_linear_model # function that builds linear model
    elif args.build_fn == "nonlinear":
        build_fn = build_nonlinear_model # function that builds non-linear model       
    else:
        print "Invalid build function name {}".format(args.build_fn)
        sys.exit(1)


     # ---- lines to be added ----
     # load model (and thus, ignore prior build function)
    if len(args.load_model) > 0:
        build_fn = lambda x: tf.keras.models.load_model(args.load_model, compile=False)
    # ---- end of lines to be added ----


    # run the main function
    main(args.n, args.epochs, args.lr, args.visualize_training_data, build_fn=build_fn, batch_size=args.batch_size)
    sys.exit(0)