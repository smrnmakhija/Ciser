import random as rnd
import numpy
#bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5

BET_MIN=1
BET_MAX=1.01
EPS_MIN=0.1
EPS_MAX=1
RHO_MIN=0.9
RHO_MAX=0.95
MU_MIN=5.4795e-5
MU_MAX=6.0883e-5
GAM_MIN=0.0714
GAM_MAX=0.1
SIG_MIN=0.0357
SIG_MAX=0.1429
TAW_MIN=9.1324e-4
TAW_MAX=0.0014
OMEG_MIN=0.0333
OMEG_MAX=0.0588

dataset = []

for bet in numpy.arange(BET_MIN, BET_MAX, 0.01):
	for eps in numpy.arange(EPS_MIN, EPS_MAX, 0.01):
		for ro in numpy.arange(RHO_MIN, RHO_MAX, 0.001):
			for mu in numpy.arange(MU_MIN, MU_MAX, 0.0001):
				for gam in numpy.arange(GAM_MIN, GAM_MAX, 0.001):
					for sigma in numpy.arange(SIG_MIN, SIG_MAX, 0.001):
						for taw in numpy.arange(TAW_MIN, TAW_MAX, 0.0001):
							for wa in numpy.arange(OMEG_MIN, OMEG_MAX, 0.01):
								R0=bet*sigma*((taw+mu)+eps*ro*gam)/((sigma+mu)*(gam+mu)*(taw+mu))
								#if R0 < 1:
								print(bet, ",", eps, ",", ro, ",", mu, ",", gam, ",", sigma, ",", taw, ",",wa, ",", R0)
								datarow = [bet, eps, ro, mu, gam, sigma, taw, R0]
								dataset.append(datarow)
