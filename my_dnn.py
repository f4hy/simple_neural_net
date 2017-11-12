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
    return (r,grad)


transfer_functions = {"fahy": fahy, "reLU": reLU, "tanh": tanh, "sigmoid": sigmoid}
search_methods = ["normal", "adagrad"]


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

        #self.weights = np.random.normal(loc=0.0, scale=1/np.sqrt(inputSize), size=(inputSize, outputSize))
        self.weights = np.random.normal(loc=0.0, scale=0.5, size=(inputSize, outputSize))
        print self.weights
        self.biasFactor = 0.0 if nobias else 1.0
        self.biasWeight = np.random.normal(loc=1.0, scale=0.01, size=(1, outputSize))*self.biasFactor
        #self.biasWeight = 0.0
        # self.weights = 2.5*np.random.random(size=(inputSize, outputSize))
        self.gradFromLastInput = None
        self.lastInput = None
        self.prevWeightDelta = []
        self.lastMeanError = 0.0

    def runForward(self, inputs):
        # logging.debug("{}weights =\n {}".format(self.weights.shape, self.weights))
        #logging.debug("{}biasweights =\n {}".format(self.biasWeight.shape, self.biasWeight))
        dot = np.dot(inputs, self.weights) + self.biasWeight * self.biasFactor
        # logging.debug("dot={}".format(dot))
        result, self.gradFromLastInput = self.activation(dot)
        # logging.debug("result={}".format(result))
        self.lastInput = inputs
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
            self.weights += weightChange * (self.learningRate/self.runCount)
            self.biasWeight += biasChange*self.biasFactor * 0.1 * (self.learningRate/self.runCount)
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

    def __init__(self, input_size, output_size, hidden_sizes=[16,16], activation=tanh, learningRate=1.0, nobias=False, outfile=None, method="normal"):

        # input_size = len(training_data[0])
        # output_size = len(correct_results[0])

        sizes = [input_size] + hidden_sizes + [output_size]
        logging.debug(sizes)
        print zip(sizes, sizes)
        print pairwise(sizes)
        for p in pairwise(hidden_sizes):
            print p

        input_layer = NeuralNetworkLayer(input_size, hidden_sizes[0], activation=activation, learningRate=learningRate,
                                         method=method, nobias=True)
        output_layer = NeuralNetworkLayer(hidden_sizes[-1], output_size, activation=activation, learningRate=learningRate,
                                          method=method)
        hidden_layers = [NeuralNetworkLayer(in_size, out_size, activation=activation, learningRate=learningRate,
                                            method=method)
                         for (in_size, out_size) in pairwise(hidden_sizes)]

        self.layers = [input_layer] + hidden_layers + [output_layer]

    def run_forward(self, input_data):
        i = input_data
        for layer in self.layers:
            o = layer.runForward(i)
            i = o
        return o

    def back_prop(self, error):
        e = error
        for layer in reversed(self.layers):
            o = layer.backProp(e)
            e = o


def mnist_digits():
    """ classify digits """
    inputs, results =  mnist_loader.load_data_wrapper()
    return (inputs, results)


def softmax(v, theta=2.0):
    # theta = 5.0
    y = np.abs(v)*theta
    m = max(y)
    v_exp = np.exp((y-m))
    return v_exp / np.sum(v_exp)


def evaluate_classifier(guess_array, correct_array, print_sample=False):
    guess = np.argmax(guess_array, axis=1)
    correct = np.argmax(correct_array, axis=1)
    num_correct = np.count_nonzero(guess == correct)
    percent_correct = 100*num_correct/len(guess)
    if print_sample:
        logging.info("Correct:{}/{} [{:.0f}%]".format(num_correct, len(guess), 100*num_correct/len(guess)))
        for i in range(min(len(guess), 6)):
            confidence = np.max(softmax(guess_array[i]))
            # g= np.clip(guess_array[i], 0, 1)
            # s = np.sum(g)
            # confidence = np.max(g/s)
            # logging.debug("softmax {}, maxsoftmax {:.2f}, max normed {:.2f}".format(softmax(g), np.max(softmax(g)), confidence))
            logging.info("Output: {}, guess:{} correct:{}, confidence {:.0f}%".format(guess_array[i], guess[i], correct[i], 100*confidence))
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
    print inputv.shape
    print y.shape
    exit()
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
    return (inputv, y)


def run_network(options, outfile):
    """ DESCRIPTION """
    logging.debug("Called with {}".format(options))

    transfer_fun = transfer_functions[options.transferfunction]

    # Array of input training cases and their correct answers
    # inputv, truthv = square_classifier()
    inputv, truthv = mnist_digits()
    # inputv,truthv = binary_xor()

    nn = NeuralNetwork(inputv.shape[1], truthv.shape[1], hidden_sizes=[30], activation=transfer_fun, learningRate=options.learningrate,
                       method=options.method, nobias=True)

    printEveryN = 5
    trainingSetSize = len(truthv)
    for j in range(20000):
        if (j % printEveryN) == 0:
            indicies = np.arange(trainingSetSize)
        else:
            indicies = np.random.choice(trainingSetSize, trainingSetSize)
        x = inputv[indicies]
        y = truthv[indicies]
        output = nn.run_forward(x)
        error = y - output
        if (j % printEveryN) == 0:
            evaluate_classifier(output, y, print_sample=True)
            logging.info("Error_{}:{}".format(j, np.mean(np.abs(error))))
        nn.back_prop(error)
    exit(-1)

    # logging.info("inputv\n{}".format(inputv))
    # logging.info("truthv\n{}".format(truthv))
    # indicies = np.random.choice(len(truthv), len(truthv)-1 )
    # logging.info("indixies={}".format(indicies))
    # logging.info("inputv\n{}".format(inputv[indicies]))
    # logging.info("truthv\n{}".format(truthv[indicies]))
    # #exit()

    # logging.info("inputv {}, inputv.T {}".format(inputv, inputv.T))
    # logging.info("inputv.shape {}, inputv.T.shape {}".format(inputv.shape, inputv.T.shape))
    # # logging.info("reLU({})={}".format(inputv, reLU(inputv)))
    # # logging.info("sigmoid({})={}".format(inputv, sigmoid(inputv)))


    # n1 = NeuralNetworkLayer(784, 16, activation=transfer_fun, learningRate=options.learningrate,
    #                         method=options.method, nobias=True)
    # n2 = NeuralNetworkLayer(16, 16, activation=transfer_fun, learningRate=options.learningrate,
    #                         method=options.method)
    # n3 = NeuralNetworkLayer(16, 10, activation=transfer_fun, learningRate=options.learningrate,
    #                         method=options.method)

    # outfile = open(outfile, 'w')
    # outfile.write("#gen,error")

    # trainingSetSize = len(truthv)

    # printEveryN = 5


    # for j in range(60000):
    #     if (j % printEveryN) == 0:
    #         indicies = np.arange(trainingSetSize)
    #     else:
    #         indicies = np.random.choice(trainingSetSize, trainingSetSize)
    #     x = inputv[indicies]
    #     y = truthv[indicies]

    #     o1 = n1.runForward(x)
    #     o2 = n2.runForward(o1)
    #     o3 = n3.runForward(o2)
    #     error = y - o3
    #     if (j % printEveryN) == 0:
    #         # logging.info("biases, n1 adagrad{} biasadagrad{}\n{}".format(n1.adagrad, n1.biasadagrad, n1.biasWeight))
    #         # logging.info("biases, n2 adagrad{} biasadagrad{}\n{}".format(n2.adagrad, n2.biasadagrad, n2.biasWeight))
    #         # logging.info("biases, n3 adagrad{} biasadagrad{}\n{}".format(n3.adagrad, n3.biasadagrad, n3.biasWeight))
    #         # logging.info("output{}\n {}".format(j,o3))
    #         evaluate_classifier(o3, y, print_sample=True)
    #         logging.info("Error_{}:{}".format(j, np.mean(np.abs(error))))
    #         outfile.flush()
    #     correct = evaluate_classifier(o3, y)
    #     outfile.write("{},{},{}\n".format(j, str(np.mean(np.abs(error), dtype=np.float64)), correct))
    #     b3 = n3.backProp(error)
    #     b2 = n2.backProp(b3)
    #     b1 = n1.backProp(b2)

    # outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple Neural Network Example")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-r", "--random", type=int,default=1,
                        help="randomseed")
    parser.add_argument("-t", "--transferfunction", type=str,default="tanh", choices=transfer_functions.keys(),
                        help="select a transfer function to use for the network")
    parser.add_argument("-m", "--method", type=str, default="normal", choices=search_methods,
                        help="method to search for minimum")
    parser.add_argument("-l", "--learningrate", type=float, default=0.8,
                        help="base learning rate")
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
    outfile = "{}_{}_{}_{}.out".format(args.outstub, args.transferfunction, args.learningrate, args.method)
    run_network(args, outfile)
