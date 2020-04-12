import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def form_nn_layers(layer_dictionary):
    layers_number = len(layer_dictionary)
    param_value = {}
    for i, layer in enumerate(layer_dictionary):
        lay_index = i+1
        lay_inpsize = layer["input_dim"]
        lay_outsize= layer["output_dim"]
        print('w for weights and b for bias')
        param_value['w' + str(lay_index)] = np.random.randn(lay_inpsize, lay_outsize)*0.05
        param_value['b' + str(lay_index)] = np.random.randn(lay_outsize, 1)*0.05

    return param_value

def feed_forward_single(input, weights, bias):
    total=0
    # inp = input.values.T.tolist()
    # wei = weights.values.T.tolist()
    for i in range(len(input)):
        for j in range(len(weights)):
            total+=input[i]*weights[j]
    # total = np.dot(weights, input) + bias
    return sigmoid(total+bias), total

def full_forward(X, params, nn_diction):
    store = {}
    inp_current = X
    for i, layer in enumerate(nn_diction):
        lay_index = i+1
        inp_prev = inp_current
        weight_current = params["w"+str(lay_index)]
        bias_curr = params["b"+str(lay_index)]
        inp_current,z_curr = feed_forward_single(inp_prev, weight_current, bias_curr)

        store["inp" + str(i)]=inp_prev
        store["z"+str(lay_index)]=z_curr
    return inp_current,store

def get_cost_value(Yprime,Y):
        m = Yprime.shape[1]
        cost = -1/m*(np.dot(Y,np.log(Yprime).T) + np.dot(1-Y, np.log(1-Yprime).T))
        return np.squeeze(cost)

def get_accuracy(Yprime,Y):
    recall_vaalue = 0
    # recall= true positive/true positive + false negative

def main_func(X,Y,batches, LR):
    nn_dict = [{"input_dim": 0, "output_dim": 0, "activation":"sigmoid"}]

    param = form_nn_layers(nn_dict)
    costs= []
    accuracies = []
    for i in range(batches):
        Yprime, c = full_forward(X,param,nn_dict)
        this_cost = get_cost_value(Yprime, Y)
        this_accuracy = get_accuracy(Yprime,Y)

        costs.append(this_cost)

    return costs

musicdata = pd.read_csv('sample-data.csv')
X = musicdata.iloc[:, :-1].values
Y = musicdata.iloc[:, -1].values
# X = pd.DataFrame(X)
# Y = pd.DataFrame(Y)

print(main_func(X,Y,batches=10,LR=0.01))