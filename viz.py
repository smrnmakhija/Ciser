import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from numpy.polynomial import polynomial as P
import pylab
from scipy.optimize import curve_fit
from math import log
from math import sqrt
import numpy.polynomial.polynomial as poly

R=0
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.001 , 0.1 , 0.901 , 5.4795e-05 , 0.0874 , 0.0747 , 0.00131324 , 0.0333 , 0.078#Case 1
bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.01 , 0.1 , 0.901 , 5.4795e-05 , 0.0744 , 0.0647 , 0.00111324 , 0.0533 , 0.90#Case 2
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.1 , 0.1 , 0.901 , 5.4795e-05 , 0.0744 , 0.1247 , 0.00091324 , 0.0333 , 10.64#Case 3
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.05 , 0.1 , 0.9 , 5.4795e-05 , 0.0974 , 0.0567 , 0.00121324 , 0.0433 , 4.056#Case 4
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.2 , 0.1 , 0.9 , 5.4795e-05 , 0.0954 , 0.1237 , 0.00111324 , 0.0533 , 17.489#Case 5
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.4 , 0.1 , 0.9 , 5.4795e-05 , 0.0894 , 0.1017 , 0.00091324 , 0.0333 , 41.615#Case 6
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.6 , 0.1 , 0.905 , 5.4795e-05 , 0.0714 , 0.1187 , 0.00091324 , 0.0333 , 64.417#Case 7
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 0.8 , 0.1 , 0.902 , 5.4795e-05 , 0.0994 , 0.0987 , 0.00101324 , 0.0333 , 75.528#Case 8
#bet,eps,ro,mu,gamma,sigma,taw, wa, R0 = 1.0 , 0.1 , 0.902 , 5.4795e-05 , 0.0824 , 0.0617 , 0.00091324 , 0.0533 , 105.151#Case 9
a=np.array([bet,eps,ro,gamma,sigma,wa,taw])
def fun(t, x):
	y=np.zeros((4, 1))
	#x=np.array([S,E,I,C])
	y[0]=-(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]+a[5]*R
	y[1]=(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]-a[4]*x[1]
	y[2]=a[4]*x[1]- a[3]*x[2]
	y[3]=a[2]*a[3]*x[2]-a[5]*x[3]
	#y[4]=(x[2]-a[2])*a[3]*x[2]+a[6]*x[3]-a[5]*x[4]
	return(y)

def diagonal(matrix):
    return sum([matrix[i][i] for i in range(min(len(matrix[0]),len(matrix)))])

solver = ode(fun)
solver.set_integrator('dopri5')

t0 = 0.0
x0 = [0.86, 0.03,0.02,0.01]
if(sum(x0)<1):

	solver.set_initial_value(x0, t0)


	t1 = 2*365
	N = t1*2
	t = np.linspace(t0, t1, N)
	sol = np.empty((N, 4))
	sol[0] = x0


	k = 1
	while solver.successful() and solver.t < t1:
    		solver.integrate(t[k])
    		sol[k] = solver.y
    		k += 1



S=sol[:,0]
E=sol[:,1]
I=sol[:,2]
C=sol[:,3]


Inf,cov = np.polyfit(t,I, 3, cov="True")

f = np.poly1d(Inf)
print f
print sqrt(diagonal(cov))
