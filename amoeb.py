import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


def fun(t, x):
	y=np.zeros((4, 1))
	bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5
	a=np.array([bet,eps,ro,gam,sigma,wa,taw,mu])
	#x=np.array([S,E,I,C])
	y[0]=(a[7]+a[5])-(a[0]*x[2]+a[2]*a[0]*x[3]+(a[7]+a[5]))*x[0]-a[5]*(x[1]+x[2]+x[3])
	y[1]=(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]-(a[4]+a[7])*x[1]
	y[2]=a[4]*x[1]-(a[3]+a[7])*x[2]
	y[3]=a[2]*a[3]*x[2]-(a[5]+a[7])*x[3]
	#print(y)
  	return y





bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5
a=np.array([bet,eps,ro,gam,sigma,wa,taw,mu])
A=np.zeros((4, 1))
R0=bet*sigma*((taw+mu)+eps*ro*gam)/((sigma+mu)*(gam+mu)*(taw+mu))
print R0
if(R0>1):
	D=(a[4]+a[5]+a[7])*(a[3]+a[7]/a[4]+a[5]*(1+a[2]*a[3])/(a[6]+a[7]))
	A[2]=Ieq=(a[5]+a[7])/(D*(1-1/R0))
	A[1]=Eeq=(a[3]+a[7])/a[4]*Ieq
	A[3]=Ceq=a[2]*a[3]*Ieq/(a[6]+a[7])
	A[0]=Seq=1/R0
else:
	Seq=1
	Ieq=0
	Eeq=(a[3]+a[7])/a[4]*Ieq
	Ceq=(a[2]*a[3]*Ieq)/(a[6]+a[7])

print Seq,Eeq,Ieq,Ceq
t0 = 0.0
x0 = [0.86, 0.03,0.02,0.01]

solver = ode(fun)
solver.set_integrator('dopri5', rtol=1e-6)

t0 = 0.0
x0 = [0.86, 0.03,0.02,0.01]
if(sum(x0)<1):

	solver.set_initial_value(x0, t0)
	t1 = 3*365
	N = 75
	t = np.linspace(t0, t1, N)
	sol = np.empty((N, 4))
	sol[0] = x0


	k = 1
	while solver.successful() and solver.t < t1:
    		solver.integrate(t[k])
    		sol[k] = solver.y
    		k += 1
#print(sol)

S=sol[:,0]
E=sol[:,1]
I=sol[:,2]
C=sol[:,3]
"""
plt.plot(t, sol[:,0], label='S')
plt.plot(t, sol[:,1], label='E')
plt.plot(t, sol[:,2], label='I')
plt.plot(t, sol[:,3], label='C')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()
"""
#R=1-(S+E+I+C)

