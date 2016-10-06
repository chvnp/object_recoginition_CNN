#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import cPickle
import copy
import math
import numpy as np
import os
import time

import cv2
import lasagne
import theano
import theano.tensor as T

imgstack = []
imgstack2 = []
labels = []
dataset1 = np.empty((364, 636), int)
dataset2 = np.empty((364, 636), int)


def load_dataset():
    folder1 = "./train/neg"  # labesl= 0
    folder2 = "./train/positives"  # label = 1


    for filename in os.listdir(folder1):
        img = cv2.imread(os.path.join(folder1, filename))
        # print(filename)
        # print(img.shape)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(img.shape)
        img = np.asarray(img, dtype='float64') / 255.
        img_ = img.reshape(-1, 1, 60, 60)

        imgstack.append(img_)
        # stack = np.append(stack,img,axis=0)
        labels.append(0)

    print("###################")
    for filename in os.listdir(folder2):
        # print(filename)
        # print(img.shape)

        img = cv2.imread(os.path.join(folder2, filename))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.asarray(img, dtype='float64') / 255.

        # put image in 4D tensor of shape (1, 3, height, width)
        img_ = img.reshape(-1, 1, 60, 60)
        imgstack.append(img_)
        labels.append(1)

    X = np.concatenate(imgstack)

    Y = np.asarray(labels).astype(np.int32)
    print(X.shape)
    return X, Y


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 60, 60),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=6, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 144 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=1728,
        nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 144 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=864,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main(model='mlp', num_epochs=2900):
    # Load the dataset
    print("Loading data...")
    X_train, Y_train = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.06, momentum=0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    # for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    best_error = 5
    # for iCount in range(0,len(X_train)):
    for epoch in range(num_epochs):
        if best_error < 0.08:
            break
        print(epoch)
        inputs = X_train
        print(X_train.shape)
        targets = Y_train
        train_err = train_fn(inputs, targets)
        if not math.isnan(train_err):
            print(train_err)
            if best_error > train_err:
                best_error = train_err
                save_model = copy.deepcopy(network)
                print("Saving the Network")
        train_batches = 1
    print("Total time taken this loop: ", time.time() - start_time)
    f = file('trained_model.save', 'wb')

    cPickle.dump(save_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


main()
