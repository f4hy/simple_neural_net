#!/usr/bin/env python2
import logging
import argparse
import numpy as np


def sigmoid(x, grad=False):
    if grad:
        output = sigmoid(x, False)
        return output*(1.0 - output)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def tanh(x, grad=False):
    if grad:
        output = tanh(x, False)
        return (1.0 - output**2)
    else:
        return np.tanh(x)

def fahy(x, grad=False):
    linear = 0.01
    if grad:
        output = fahy(x, False)
        return (1.0 - output**2) + linear
    else:
        return np.tanh(x) + linear*x



def reLU(x, grad=False):
    if grad:
        return np.piecewise(x, [x < 0., x >= 0.], [0., lambda x: 1.])
    else:
        return np.piecewise(x, [x < 0., x >= 0.], [0., lambda x: x])


transfer_functions = {"fahy": fahy, "reLU": reLU, "tanh": tanh, "sigmoid": sigmoid}

class NeuralNetworkLayer(object):
    """ A single layer of a neural network"""

    def __init__(self, inputSize, outputSize, activation=reLU, learningRate=1.0, outfile=None):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.learningRate = learningRate
        self.adagrad = 0.0


        self.outfile = None
        if outfile is not None:
            self.outfile = open(outfile,'w')



        self.weights = np.random.normal(loc=0.0, scale=0.25, size=(inputSize, outputSize))
        # self.weights = 2.5*np.random.random(size=(inputSize, outputSize))
        self.gradFromLastInput = None
        self.lastInput = None
        self.prevWeightDelta = []
        self.lastMeanError = 0.0

    def runForward(self, inputs):
        logging.debug("{}weights =\n {}".format(self.weights.shape, self.weights))
        dot = np.dot(inputs, self.weights)
        logging.debug("dot={}".format(dot))
        result = self.activation(dot)
        logging.debug("result={}".format(result))
        self.lastInput = inputs
        self.gradFromLastInput = self.activation(result, grad=True)
        logging.debug("grad={}".format(self.gradFromLastInput))
        return result

    def backProp(self, errors):
        logging.debug("Backproping errors {}".format(errors))
        mydelta = errors * self.gradFromLastInput
        logging.debug("mydelta={}".format(mydelta))
        mygrad = np.dot(mydelta, self.weights.T)
        logging.debug("mygrad={}".format(mygrad))
        logging.debug("lastInput={}".format(self.lastInput))
        logging.debug("lastInputShape {} mydelta shape {}".format(self.lastInput.shape, mydelta.shape))
        logging.debug("lastInput^t Shape {} mydelta shape {}".format(self.lastInput.T.shape, mydelta.shape))
        weightChange = (self.lastInput.T).dot(mydelta)


        self.adagrad += weightChange**2
        logging.debug("adagrad {}".format(self.adagrad))
        self.weights += weightChange*self.learningRate / (np.sqrt(self.adagrad) + 1e-7)
        logging.debug("weight Change\n {}".format(weightChange))
        logging.debug("new weights\n {}".format(self.weights))

        self.learningRate *= 0.9999

        if self.outfile is not None:
            self.outfile.write("{}".format(self.weights.flat))

        meanError = np.mean(np.abs(mygrad))

        return mygrad


def binary_xor():
    """ Classify xor with an extra bit which needs to be ignored"""
    # Array of input training cases
    inputv = np.array([[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [0, 0, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 0]])
    # Expected result vectors used to train the network
    y = np.array([[0],
                  [1],
                  [1],
                  [0],
                  [0],
                  [1],
                  [1],
                  [0]])
    return (inputv, y)

def run_network(options):
    """ DESCRIPTION """
    logging.debug("Called with {}".format(options))

    transfer_fun = transfer_functions[options.transferfunction]

    # Array of input training cases and their correct answers
    inputv,y = = binary_xor()

    logging.info("inputv {}, inputv.T {}".format(inputv, inputv.T))
    logging.info("inputv.shape {}, inputv.T.shape {}".format(inputv.shape, inputv.T.shape))
    # logging.info("reLU({})={}".format(inputv, reLU(inputv)))
    # logging.info("sigmoid({})={}".format(inputv, sigmoid(inputv)))

    n1 = NeuralNetworkLayer(3, 6, activation=transfer_fun, learningRate=1.05)
    n2 = NeuralNetworkLayer(6, 4, activation=transfer_fun, learningRate=1.05)
    n3 = NeuralNetworkLayer(4, 1, activation=transfer_fun, learningRate=1.05)

    outfile = open(options.outfile,'w')
    outfile.write("#gen,error")

    for j in range(6000):

        o1 = n1.runForward(inputv)
        o2 = n2.runForward(o1)
        o3 = n3.runForward(o2)
        error = y - o3
        if (j % 100) == 0:
            logging.info("output{}\n {}".format(j,o3))
            logging.info("error\n {}".format(error))
            logging.info("Error:" + str(np.mean(np.abs(error))))
        outfile.write("{},{}\n".format(j,str(np.mean(np.abs(error)))))
        b3 = n3.backProp(error)
        b2 = n2.backProp(b3)
        b1 = n1.backProp(b2)

    outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple Neural Network Example")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-r", "--random", type=int,default=1,
                        help="randomseed")
    parser.add_argument("-t", "--transferfunction", type=str,default="tanh", choices=transfer_functions.keys(),
                        help="select a transfer function to use for the network")
    parser.add_argument("-o", "--outfile", type=str,
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


    run_network(args)
