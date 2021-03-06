# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:03:18 2015

@author: alym
"""

###################
# Update items below for each train/test
###################

# training params
epochs=200
step=1e-3
wvecDim=30
rho = 1e-4

# for RNN2 only, otherwise doesnt matter
middleDim=25

model="RNTN" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if model == "RNN2":
    outfile = "models/%s_wvecDim_%d_middleDim_%d_step_%f_2.bin" % (model, 
        wvecDim, middleDim, step)

else:
    outfile = "models/%s_wvecDim_%d_step_%f_2_rho_%f.bin" % (model, wvecDim, step, rho)
    


print outfile

import runNNet
args = "--step %f --epochs %d --outFile %s --middleDim %d --outputDim %d --wvecDim %d --rho %f --model %s" % \
    (step, epochs, outfile, middleDim, 5, wvecDim, rho, model) 
runNNet.run(args.split(' '))

