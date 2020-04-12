import numpy as np
import pandas as pd

# module for computing sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# module for forming the given layers
def form_nn_layers(layer_dictionary):
    # get number of layers needed
    layers_number = len(layer_dictionary)

    # we need a memory to store the parameters information so param_value is a dictionar
    param_value = {}
    for i, layer in enumerate(layer_dictionary):
        lay_index = i+1

        # get input and output dimension and store in corresponding variables
        lay_inpsize = layer["input_dim"]
        lay_outsize= layer["output_dim"]

        print('w for weights and b for bias')
        param_value['w' + str(lay_index)] = np.random.randn(lay_inpsize, lay_outsize)*0.05
        param_value['b' + str(lay_index)] = np.random.randn(lay_outsize, 1)*0.05

    return param_value

# forward propogation for single layer module that can be used for finding full forward propogation
def feed_forward_single(input, weights, bias):
    total = np.dot(weights, input) + bias
    return sigmoid(total+bias), total

# full forward propogation module for given NN parameters and architecture
def full_forward(X, params, nn_diction):
    # dictionary to store the values of input and z at ech iteration
    store = {}
    inp_current = X
    # for each index and corresponding layer of the network,  single layer feed forward is called and values get updated
    # storing in 'stores'
    for i, layer in enumerate(nn_diction):
        lay_index = i+1
        inp_prev = inp_current
        weight_current = params["w"+str(lay_index)]
        bias_curr = params["b"+str(lay_index)]

        # invoke single layer forward propogation with existing values of input, weight and bias.
        inp_current,z_curr = feed_forward_single(inp_prev, weight_current, bias_curr)

        # append updated values to the 'store'
        store["inp" + str(i)]=inp_prev
        store["z"+str(lay_index)]=z_curr
    return inp_current,store

# loss function to get cost module
def get_cost_value(Yprime,Y):
        m = Yprime.shape[1]
        cost = -1/m*(np.dot(Y,np.log(Yprime).T) + np.dot(1-Y, np.log(1-Yprime).T))
        return np.squeeze(cost)

# module to get accuracy
def get_accuracy(Yprime,Y):
    # recall can be used as an accuracy measure..
    recall_vaalue = 0
    # recall= true positive/true positive + false negative
    return recall_vaalue


def main_func(X,Y,batches, LR):
    # dictionary to store architecture of NN so as to create layers
    nn_dict = [{"input_dim": 100, "output_dim": 25, "activation":"sigmoid"}]

    # parameters of the NN
    param = form_nn_layers(nn_dict)

    # cost function (loss function) stored
    costs= []
    accuracies = []
    for i in range(batches):
        Yprime, c = full_forward(X,param,nn_dict)
        # get cost from cost module
        this_cost = get_cost_value(Yprime, Y)
        # get accuracy form accuracy module (not yet completed)...
        this_accuracy = get_accuracy(Yprime,Y)
        costs.append(this_cost)

        # once accuracy is found, it too should be returned by this functionality
    return costs

# get the split data from the module from read_preprocess_data
musicdata = pd.read_csv('sample-data.csv')
X = musicdata.iloc[:, :-1].values
Y = musicdata.iloc[:, -1].values
# X = pd.DataFrame(X)
# Y = pd.DataFrame(Y)

# invoke the main functionality to neural network
# - 'batches' is number of times training vector is used and 'LR' is learning rate (can be modified for each try)
print(main_func(X,Y,batches=10,LR=0.01))