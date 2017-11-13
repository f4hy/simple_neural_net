#!/usr/bin/env python2
import logging
import argparse
import numpy as np
import cPickle
import gzip
import mnist_loader
from itertools import tee, izip


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    grad = s*(1.0-s)
    return (s, grad)


def tanh(x):
    t = np.tanh(x)
    grad = (1.0 - t**2)
    return (t, grad)


def fahy(x):
    linear = 0.005
    t = np.tanh(x)
    f = t + linear*x
    g = (1.0 - t**2) + linear
    return (f, g)


def reLU(x):
    r = np.piecewise(x, [x < 0., x >= 0.], [0., lambda x: x])
    grad = np.piecewise(x, [x < 0., x >= 0.], [0., lambda x: 1.])
    return (r, grad)




class NeuralNetworkLayer(object):
    """ A single layer of a neural network"""

    def __init__(self, inputSize, outputSize, activation=reLU, learningRate=1.0, nobias=False, outfile=None, method="normal"):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.learningRate = learningRate
        self.adagrad = 1.0
        self.biasadagrad = 1.0
        self.method = method
        self.runCount = 1

        self.outfile = None
        if outfile is not None:
            self.outfile = open(outfile, 'w')

        # self.weights = np.random.normal(loc=0.0, scale=1/np.sqrt(inputSize), size=(inputSize, outputSize))
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(inputSize, outputSize))
        self.biasFactor = 0.0 if nobias else 1.0
        self.biasWeight = np.random.normal(loc=1.0, scale=0.01, size=(1, outputSize))*self.biasFactor
        self.gradFromLastInput = None
        self.lastInput = None
        self.prevWeightDelta = []
        self.lastMeanError = 0.0

    def runForward(self, inputs, training=True):
        # logging.debug("{}weights =\n {}".format(self.weights.shape, self.weights))
        #logging.debug("{}biasweights =\n {}".format(self.biasWeight.shape, self.biasWeight))
        dot = np.dot(inputs, self.weights) + self.biasWeight * self.biasFactor
        # logging.debug("dot={}".format(dot))
        result, grad = self.activation(dot)
        # logging.debug("result={}".format(result))
        if training:
            self.lastInput = inputs
            self.gradFromLastInput = grad
        # logging.debug("grad={}".format(self.gradFromLastInput))
        return result

    def backProp(self, errors):
        # logging.debug("Backproping errors {}".format(errors))
        mydelta = errors * self.gradFromLastInput
        # logging.debug("mydelta={}".format(mydelta))
        mygrad = np.dot(mydelta, self.weights.T)
        # logging.debug("mygrad={}".format(mygrad))
        # logging.debug("lastInput={}".format(self.lastInput))
        # logging.debug("lastInputShape {} mydelta shape {}".format(self.lastInput.shape, mydelta.shape))
        # logging.debug("lastInput^t Shape {} mydelta shape {}".format(self.lastInput.T.shape, mydelta.shape))
        #logging.debug("biasWeight Shape {}, ones shape {}".format(self.biasWeight.shape, (np.ones((1, self.lastInput.shape[1])).shape)))
        weightChange = (self.lastInput.T).dot(mydelta)
        biasChange = (np.ones((1,self.lastInput.shape[0]))).dot(errors)
        # biasChange = mygrad
        # logging.debug("old weights\n {}".format(self.weights))

        if self.method == "normal":
            self.weights += weightChange * (self.learningRate)
            self.biasWeight += biasChange*self.biasFactor * (self.learningRate)
        if self.method == "decay":
            self.weights += weightChange * (self.learningRate/self.runCount)
            self.biasWeight += biasChange*self.biasFactor * (self.learningRate/self.runCount)
            self.runCount += 0.001
        elif self.method == "adagrad":
            self.adagrad += weightChange**2
            # self.adagrad = 0.999*self.adagrad + (0.1) * weightChange**2
            self.biasadagrad += biasChange**2
            self.weights += weightChange*self.learningRate / (np.sqrt(self.adagrad) + 1e-7)
            self.biasWeight += biasChange*self.learningRate / (np.sqrt(self.biasadagrad) + 1e-7)
        else:
            raise "Unsuported"
        # logging.debug("weight Change\n {}".format(weightChange))
        # logging.debug("new weights\n {}".format(self.weights))
        # # exit()
        # #self.learningRate *= 0.999

        # if self.outfile is not None:
        #     self.outfile.write("{}".format(self.weights.flat))

        meanError = np.mean(np.abs(mygrad))

        return mygrad


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


class NeuralNetwork(object):
    """ A single layer of a neural network"""

    def __init__(self, input_size, output_size, hidden_sizes=[30], activation=tanh, learningRate=1.0, nobias=False, outfile=None, method="normal"):
        logging.info("Creating a network with intput size {}, hidden sizes {}, and output size {}".format(input_size, hidden_sizes, output_size))
        sizes = [input_size] + hidden_sizes + [output_size]
        logging.debug(sizes)
        logging.info("connections have input,output sizes {}".format(list(pairwise(sizes))))

        input_layer = NeuralNetworkLayer(input_size, hidden_sizes[0], activation=activation, learningRate=learningRate,
                                         method=method, nobias=True)
        output_layer = NeuralNetworkLayer(hidden_sizes[-1], output_size, activation=activation, learningRate=learningRate,
                                          method=method)
        hidden_layers = [NeuralNetworkLayer(in_size, out_size, activation=activation, learningRate=learningRate,
                                            method=method)
                         for (in_size, out_size) in pairwise(hidden_sizes)]

        self.layers = [input_layer] + hidden_layers + [output_layer]

    def run_forward(self, input_data, training=True):
        i = input_data
        for layer in self.layers:
            o = layer.runForward(i, training=training)
            i = o
        return o

    def back_prop(self, error):
        e = error
        for layer in reversed(self.layers):
            o = layer.backProp(e)
            e = o


def mnist_digits():
    """ classify digits """
    inputs, results = mnist_loader.load_data_wrapper()
    return (inputs, results)


def softmax(v, theta=2.0):
    y = v*theta
    m = np.max(y)
    v_exp = np.exp((y-m))
    return v_exp / np.sum(v_exp)


def confidence(v):
    s = np.sum(v)
    if 1.2 > s > 0.8:
        return np.max(v)/np.sum(v)
    else:
        return max(softmax(v))


def evaluate_classifier(guess_array, correct_array, print_sample=False):
    guess = np.argmax(guess_array, axis=1)
    correct = np.argmax(correct_array, axis=1)
    num_correct = np.count_nonzero(guess == correct)
    percent_correct = 100*num_correct/len(guess)
    if print_sample:
        logging.info("Correct:{}/{} [{:.0f}%]".format(num_correct, len(guess), 100*num_correct/len(guess)))
        for i in range(min(len(guess), 6)):
            conf = confidence(guess_array[i])
            # g= np.clip(guess_array[i], 0, 1)
            # s = np.sum(g)
            # confidence = np.max(g/s)
            # logging.debug("softmax {}, maxsoftmax {:.2f}, max normed {:.2f}".format(softmax(g), np.max(softmax(g)), confidence))
            logging.info("Output: {}, guess:{} correct:{}, confidence {:.0f}%".format(guess_array[i], guess[i], correct[i], 100*conf))
            # logging.info("Correct:{}, correct{}".format(correct_array[i], correct[i]))
    return percent_correct



def binary_xor():
    """ Classify xor with an extra bit which needs to be ignored"""
    # Array of input training cases
    inputv = np.array([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])
    # Expected result vectors used to train the network
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return (inputv, y)

def square_classifier():
    """ Classify xor with an extra bit which needs to be ignored"""
    # Array of input training cases
    inputv = np.array([[-0.5, -0.5, -0.5, -0.5], # solid
                       [ 0.5,  0.5,  0.5,  0.5], # soild
                       [ 0.5, -0.5,  0.5, -0.5], # vertical
                       [-0.5,  0.5, -0.5,  0.5], # vertical
                       [ 0.5, -0.5, -0.5,  0.5], # diagonal
                       [-0.5,  0.5,  0.5, -0.5], # diagonal
                       [ 0.5,  0.5, -0.5, -0.5], # horizontal
                       [-0.5, -0.5,  0.5,  0.5]]) # horizontal

    # Expected result vectors used to train the network
    y = np.array([[1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]])
    n = len(y)
    indicies = np.random.choice(n, 10000)
    return (inputv[indicies], y[indicies])


data_choices = {"digits": mnist_digits, "square": square_classifier, "xor": binary_xor}
transfer_functions = {"fahy": fahy, "reLU": reLU, "tanh": tanh, "sigmoid": sigmoid}
search_methods = ["normal", "decay", "adagrad"]


def run_network(options, outfile):
    """ DESCRIPTION """
    logging.debug("Called with {}".format(options))

    transfer_fun = transfer_functions[options.transferfunction]

    # Array of input training cases and their correct answers
    data_fun = data_choices[options.trainingdataset]
    training_data, training_truth = data_fun()



    nn = NeuralNetwork(training_data.shape[1], training_truth.shape[1], hidden_sizes=options.hiddenlayers, activation=transfer_fun, learningRate=options.learningrate,
                       method=options.method)

    trainingSetSize = len(training_truth)
    outfile = open(outfile, 'w')
    outfile.write("#evals,cost,percent_correct\n")

    def evaluate(N):
        eval_x = training_data
        eval_y = training_truth
        eval_output = nn.run_forward(eval_x, training=False)
        eval_error = eval_y - eval_output
        cost = np.sum(0.5 * (eval_error)**2)
        percent_correct = evaluate_classifier(eval_output, eval_y, print_sample=True)
        logging.info("Error_{}:{}, cost {}".format(N, np.mean(np.abs(eval_error)), cost))
        outfile.write("{},{},{}\n".format(N, str(np.mean(cost, dtype=np.float64)), percent_correct))
        outfile.flush()

    # evalEveryN = 1
    evalEveryNBatch = 10000
    batchsize = options.batchsize
    evaluation_count = 0
    for epoch in range(30):
        evaluate(evaluation_count)
        for batchindex in range(0, trainingSetSize, batchsize):
            evaluation_count += batchsize
            inputv = training_data[batchindex:batchindex+batchsize]
            truthv = training_truth[batchindex:batchindex+batchsize]
            # batch_indicies = np.random.choice(trainingSetSize, trainingSetSize)
            # batch_indicies = epoch % trainingSetSize
            batch_indicies = np.random.permutation(len(truthv))
            # print np.random.choice(trainingSetSize, trainingSetSize)
            # batch_indicies = np.arange(trainingSetSize)
            batch_x = inputv[batch_indicies]
            batch_y = truthv[batch_indicies]
            batch_output = nn.run_forward(batch_x)
            if (batchindex % evalEveryNBatch) == 0 and batchindex > 0:
                evaluate(evaluation_count)
                logging.info("Evaluating epoch {} batch {}".format(epoch, batchindex))
            batch_error = batch_y - batch_output
            nn.back_prop(batch_error)
    outfile.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple Neural Network Example")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-r", "--random", type=int, default=1,
                        help="randomseed")
    parser.add_argument("-b", "--batchsize", type=int, default=10,
                        help="randomseed")
    parser.add_argument("-t", "--transferfunction", type=str, default="tanh", choices=transfer_functions.keys(),
                        help="select a transfer function to use for the network")
    parser.add_argument("-m", "--method", type=str, default="normal", choices=search_methods,
                        help="method to search for minimum")
    parser.add_argument("-d", "--trainingdataset", type=str, default="digits", choices=data_choices,
                        help="select which training data set to learn")
    parser.add_argument("-l", "--learningrate", type=float, default=0.8,
                        help="base learning rate")
    parser.add_argument("-hl", "--hiddenlayers", type=int, nargs='+', required=True,
                        help="Specify hidden layer sizes (e.g. 16 16 8 )")
    parser.add_argument("-o", "--outstub", type=str,
                        help="path for output")
    parser.add_argument('--err', nargs='?', type=argparse.FileType('w'),
                        default=None)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.err is not None:
        root = logging.getLogger()
        ch = logging.StreamHandler(args.err)
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        root.addHandler(ch)

    logging.info("Setting random seed to %s", args.random)
    np.random.seed(args.random)

    np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    outfile = "{}_{}_hl{}_bs{}_{}_{}_{}.out".format(args.outstub, args.trainingdataset, "-".join(map(str,args.hiddenlayers)), args.batchsize, args.transferfunction, args.learningrate, args.method)
    run_network(args, outfile)
