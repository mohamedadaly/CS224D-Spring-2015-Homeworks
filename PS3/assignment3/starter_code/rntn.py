import numpy as np
import collections
np.seterr(over='raise',under='raise')

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        cost = 0.0
        correct = []
        guess = []
        total = 0.0
        
        self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
        cost += (self.rho/2)*np.sum(self.V**2)

        return scale*cost,[self.dL,
                           scale*(self.dV + self.rho*self.V),
                           scale*(self.dW + self.rho*self.W), 
                           scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),
                           scale*self.dbs]

    def forwardProp(self,node, correct, guess):
        cost = total = 0.0

        # Will do it recursively starting at the root
        #
        # a leaf node? then just have a hidden layer and a softmax layer
        if node.isLeaf:
            # set h1 to its word vector
            node.hActs1 = self.L[:, node.word]
            # compute prob
            node.probs = softmax(self.Ws.dot(node.hActs1) + self.bs)
            
        # Not a leaf, so has two children
        else:
            # compute the two children first
            c1, t1 = self.forwardProp(node.left, correct, guess)
            c2, t2 = self.forwardProp(node.right, correct, guess)
            cost += c1 + c2
            total += t1 + t2

            # concatentated vector from children            
            hb = np.concatenate((node.left.hActs1, node.right.hActs1))
            # compute h1
            node.hActs1 = np.zeros_like(node.left.hActs1)
            for d in range(self.wvecDim):
                node.hActs1[d] = np.tanh(
                    hb.dot(self.V[d].dot(hb)) + self.W[d].dot(hb) + self.b[d])
            # compute prob
            node.probs = softmax(self.Ws.dot(node.hActs1) + self.bs)

        # compute cost
        cost += - np.log(node.probs[node.label])
        # fprop
        node.fprop = True
        # add correct 
        correct.append(node.label)
        # and guess (0-based labels)
        guess.append(np.argmax(node.probs))        

        return cost,total + 1


    def backProp(self,node,error=None):

        # Clear nodes
        #node.fprop = False

        # Recursive backprop from the root
        #
        # error due to softmax (delta_3)
        ds = node.probs.copy()
        ds[node.label] -= 1
        
        # Ws and bs
        self.dWs += np.outer(ds, node.hActs1)
        self.dbs += ds

        # default        
        if error is None:
            error = np.zeros((self.wvecDim,))
            
        # add the error from this step to the error passed from above
        # the error from above is already multiplied by the right matrix
        # i.e. already passed the affine transformation
        # 
        db = self.Ws.T.dot(ds) + error

        # Not leaf, update dW and db
        if not node.isLeaf:
            # pass the nonlinearity (derivative of tanh)
            db *= (1 - node.hActs1**2)
            
            # concatente left and right hidden activations
            hb = np.concatenate((node.left.hActs1, node.right.hActs1))
            
            # W and b
            self.dW += np.outer(db, hb)
            self.db += db
            for d in range(self.wvecDim):
                self.dV[d] = db[d] * np.outer(hb, hb)
        
            # back prop children
            #
            # error to children
            dbc = self.W.T.dot(db)
            for d in range(self.wvecDim):
                dbc += db[d] * (self.V[d] + self.V[d].T).dot(hb)
            # left error
            self.backProp(node.left, error = dbc[:self.wvecDim])
            # right error
            self.backProp(node.right, error = dbc[self.wvecDim:])
        # Leaf, update L
        else:
            self.dL[node.word] += db

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dW... (might take a while)"        
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
                        
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






