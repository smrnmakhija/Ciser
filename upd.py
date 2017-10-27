import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

R=0
bet,eps,ro,gam,sigma,wa,taw=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4
a=np.array([bet,eps,ro,gam,sigma,wa,taw])
def fun(t, x):
	y=np.zeros((4, 1))
	#x=np.array([S,E,I,C])
	print a[0]
	y[0]=-(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]+a[5]*R
	y[1]=(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]-a[4]*x[1]
	y[2]=a[4]*x[1]- a[3]*x[2]
	y[3]=a[2]*a[3]*x[2]-a[5]*x[3]
	#y[4]=(x[2]-a[2])*a[3]*x[2]+a[6]*x[3]-a[5]*x[4]
 	return(y)


solver = ode(fun)
solver.set_integrator('dopri5')

t0 = 0.0
x0 = [0.86, 0.03,0.02,0.01]
r=[0]
if(sum(x0)<1):

	solver.set_initial_value(x0, t0)


	t1 = 3*365
	N = 75
	t = np.linspace(t0, t1, N)
	sol = np.empty((N, 4))
	sol[0] = x0
	R=0

	
	k = 1
	while solver.successful() and solver.t < t1:
    		solver.integrate(t[k])
    		sol[k] = solver.y
    		R=1-sum(sol[k])
    		r.append(R)
    		k += 1
#print r
# Plot

plt.plot(t, sol[:,0], label='S')
plt.plot(t, sol[:,1], label='E')
plt.plot(t, sol[:,2], label='I')
plt.plot(t, sol[:,3], label='C')
plt.plot(t, r, label='R')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

