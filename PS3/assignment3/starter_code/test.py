#!/bin/bash

# training params
epochs=30
step=1e-2
wvecDim=30

# for RNN2 only, otherwise doesnt matter
middleDim=25

model="RNN" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if model == "RNN2":
    outfile = "models/%s_wvecDim_%d_middleDim_%d_step_%f_2.bin" % (model, 
        wvecDim, middleDim, step)
else:
    outfile = "models/%s_wvecDim_%d_step_%f_2.bin" % (model, wvecDim, step)
    
print outfile

# the saved outfile from training
infile = outfile

import runNNet

# test the model on test data
#python runNNet.py --inFile $infile --test --data "test" --model $model
args = "--inFile %s --test --data %s --model %s" % (infile, 'test', model)
runNNet.run(args.split(' '))

# test the model on dev data
#python runNNet.py --inFile $infile --test --data "dev" --model $model
#args = "--inFile %s --test --data %s --model %s" % (infile, 'dev', model)
#runNNet.run(args.split(' '))

# test the model on training data
#args = "--inFile %s --test --data %s --model %s" % (infile, 'train', model)
#python runNNet.py --inFile $infile --test --data "train" --model $model












