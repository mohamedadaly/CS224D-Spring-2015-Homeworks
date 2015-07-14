from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        
        #### YOUR CODE HERE ####
        # any other initialization you need
        
        # init W & U
        self.params.W = random_weight_matrix(dims[1], dims[0])
        self.params.U = random_weight_matrix(dims[2], dims[1])
        # init b1 & b2
        self.params.b1 = zeros((dims[1],))
        self.params.b2 = zeros((dims[2],))
        # init L
        self.sparams.L = wv.copy()

        #### END YOUR CODE ####



    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####        
        
        # get word vectors for the window
        x = self.sparams.L[window, :].ravel()

        ##
        # Forward propagation
        h = tanh(self.params.W.dot(x) + self.params.b1)
        y_h = softmax(self.params.U.dot(h) + self.params.b2)

        ##
        # Backpropagation
        # check details in the part1-NER notebook

        delta_2 = y_h
        delta_2[label] -= 1.0
        
        self.grads.U += outer(delta_2, h) + self.lreg * self.params.U
        self.grads.b2 += delta_2
        
        delta_1 = self.params.U.T.dot(delta_2) * (1 - h ** 2)
        
        self.grads.W += outer(delta_1, x) + self.lreg * self.params.W
        self.grads.b1 += delta_1
               
        # get the middle part of W
        mi = self.sparams.L.shape[1]
        self.sgrads.L[window[1]] = self.params.W[:,arange(mi, mi+mi)].T.dot(delta_1)

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = array(windows)[newaxis,:]


        #### YOUR CODE HERE ####
        P = zeros((windows.shape[0], self.params.U.shape[0]))
        
        # loop over rows and compute probabilities
        for n in arange(windows.shape[0]):
            # forward propagation
            #
            # get word vectors for the window
            x = self.sparams.L[windows[n,:], :].ravel()
    
            h = tanh(self.params.W.dot(x) + self.params.b1)
            y_h = softmax(self.params.U.dot(h) + self.params.b2)
            
            P[n, :] = y_h
            

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####

        # get proba
        P = self.predict_proba(windows)
        
        # get maximum in each row
        c = argmax(P, axis=1)

        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = array(windows)[newaxis,:]
            
        # compute proba
        P = self.predict_proba(windows)
        
        # cross entropy loss
        ce = -1. * sum(log(P[arange(windows.shape[0]), labels]))
        # regularizaiton loss
        reg = (self.lreg / 2.) * (sum(self.params.W ** 2) + sum(self.params.U ** 2))
        
        J = ce + reg


        #### END YOUR CODE ####
        return J