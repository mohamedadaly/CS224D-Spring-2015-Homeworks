# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:03:18 2015

@author: alym
"""

###################
# Update items below for each train/test
###################

# training params
epochs=60
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

import runNNet
args = "--step %f --epochs %d --outFile %s --middleDim %d --outputDim %d --wvecDim %d --model %s" % \
    (step, epochs, outfile, middleDim, 5, wvecDim, model) 
runNNet.run(args.split(' '))

