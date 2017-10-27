import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

#bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5
#bet,eps,ro,gam,sigma,wa,taw,mu=0,0,0.9,0.0714,0.0357,0.0333, 9.1324e-4,5.4795e-5
#bet,eps,ro,gam,sigma,wa,taw,mu=1,1,0.95,0.1,0.1429,0.0588, 0.0014,6.0883e-5
bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5
a=np.array([bet,eps,ro,gam,sigma,wa,taw])
count = 0
def fun(t, x):
	global a
	global bet
	global eps
	global ro
	global gam
	global sigma
	global wa
	global taw
	global mu
	R=0
	y=np.zeros((4, 1))
	#x=np.array([S,E,I,C])
	print a[0]
	y[0]=-(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]+a[5]*R
	y[1]=(a[0]*x[2]+a[1]*a[0]*x[3])*x[0]-a[4]*x[1]
	y[2]=a[4]*x[1]- a[3]*x[2]
	y[3]=a[2]*a[3]*x[2]-a[5]*x[3]
	#print(y)
	#y[4]=(x[2]-a[2])*a[3]*x[2]+a[6]*x[3]-a[5]*x[4]
 	return(y)


def drawing():
	global count
	global a
	global bet
	global eps
	global ro
	global gam
	global sigma
	global wa
	global taw
	global mu
	#bet,eps,ro,gam,sigma,wa,taw,mu=0.6,0.084,0.95,0.1,0.0714,0.0588, 9.1324e-4,6.8493e-5
	#a=np.array([bet,eps,ro,gam,sigma,wa,taw])
	beta=[]
	beta_1=[]
	rep_1=[]
	for k1 in np.arange(0.071468, 0.071468,0.1):
		for k2 in np.arange(0.1000685,0.1000685,0.1):
			for k3 in np.arange(9.8173e-4,9.8173e-4,0.1):
				for p in np.arange(0.00714,0.00714,0.1):
					for sigma in np.arange(0.0714,0.0714,0.1):
						#p=epsolon*rho*gamma

							B=(k1*k2*k3)/(sigma*(k3+p))
							a[0] = B

							beta.append(B)
							R0=a[0]*sigma*((taw+mu)+eps*ro*gam)/((sigma+mu)*(gam+mu)*(taw+mu))
						#print R0, B
							if(R0>2):
								count += 1
								rep_1.append(R0)
								beta_1.append(B)
								
								solver = ode(fun)
								solver.set_integrator('dopri5')

								t0 = 0.0
								x0 = [0.66, 0.03,0.02,0.01]
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
										print(sol)
    								k+=1
    								plt.plot(t, sol[:,0], color='red')
    								plt.plot(t,sol[:,1], color='blue')
    								plt.plot(t, sol[:,2], color='green')
    								plt.plot(t,sol[:,3], color='pink')
    								#plt.plot(t, r, color='violet', label='R')
    								#plt.xlabel('t')
											
	plt.grid(True)
	plt.legend()
	plt.show()

	"""
	plt.plot(beta_1,rep_1)
	plt.grid(True)
	plt.legend()
	plt.show()
	"""

drawing()
print("Number of legit cases encountered:", count)
