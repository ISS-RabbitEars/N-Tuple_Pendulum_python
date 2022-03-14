import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(thw, ti, p):
	thw_list=thw
	m,l,g=p

	th=[]
	w=[]
	for i in range(m.size):
		th.append(thw_list[2*i])
		w.append(thw_list[2*i+1])

	sub={}
	for i in range(m.size):
		sub['M%i'%i]=m[i]
		sub['L%i'%i]=l[i]
		theta_t=theta[i]
		sub[theta_t]=th[i]
		theta_dot=theta[i].diff(t,1)
		sub[theta_dot]=w[i]
	sub['g']=g

	diffeq=[]
	for i in range(m.size):	
		diffeq.append(w[i])
		theta_ddot=theta[i].diff(t,2)
		diffeq.append(solution_set[theta_ddot].subs(sub))

	print(ti)

	return diffeq	

N=3

L=sp.symbols('L0:%d'%N)
M=sp.symbols('M0:%d'%N)
g=sp.symbols('g')
t=sp.Symbol('t')
theta=dynamicsymbols('theta0:%d'%N)

x=[]
xd=[]
y=[]
yd=[]
vs=[]
for i in range(N):
	x.append(L[0]*sp.sin(theta[0]))
	y.append(-L[0]*sp.cos(theta[0]))
	for j in range(1,i+1):
		x[i]+=L[j]*sp.sin(theta[j])
		y[i]+=-L[j]*sp.cos(theta[j])
	xd.append(x[i].diff(t,1))
	yd.append(y[i].diff(t,1))
	vs.append(sp.simplify(xd[i]**2+yd[i]**2))

T=0.5*M[0]*vs[0]
V=M[0]*g*y[0]
for i in range(1,N):
	T+=0.5*M[i]*vs[i]
	V+=M[i]*g*y[i]

LG=T-V

LGdiff=[]
theta_dot=[]
theta_ddot=[]
for i in range(N):
	theta_dot.append(theta[i].diff(t,1))
	theta_ddot.append(theta[i].diff(t,2))
	dLdtheta=LG.diff(theta[i],1)
	dLdthetadot=LG.diff(theta_dot[i],1)
	ddtdLdthetadot=dLdthetadot.diff(t,1)
	LGdiff.append(sp.simplify(ddtdLdthetadot-dLdtheta))

solution_set=sp.solve(LGdiff,theta_ddot)

#-----------------------------------------------------------

gc=9.8
mass_a=1
mass_b=1
length_a=0.5
length_b=1
theta_a=90
theta_b=180
omega_a=0
omega_b=0

m=np.linspace(mass_a,mass_b,N)
l=np.linspace(length_a,length_b,N)
tho=np.linspace(theta_a,theta_b,N)
tho*=np.pi/180
wo=np.linspace(omega_a,omega_b,N)

p=[m,l,gc]

thowo=[]
for i in range(N):
	thowo.append(tho[i])
	thowo.append(wo[i])

tf = 30 
nfps = 30 
nframes = tf * nfps
tt = np.linspace(0, tf, nframes)

sol = odeint(integrate, thowo, tt, args = (p,))

tha=np.zeros((N,nframes))
wa=np.zeros((N,nframes))
for i in range(N):
	tha[i]=sol[:,2*i]
	wa[i]=sol[:,2*i+1]

xa=np.zeros((N,nframes))
ya=np.zeros((N,nframes))
for i in range(N):
	for j in range(i+1):
		xa[i]+=l[j]*np.sin(tha[j])
		ya[i]+=-l[j]*np.cos(tha[j])

ldim=l.sum()
lmax=ldim+0.2
lmin=-ldim-0.2

pea=np.zeros((N,nframes))
for i in range(N):
	pea[i]=m[i]*gc*ya[i]

kea=np.zeros((N,nframes))
for i in range(N):
	for j in range(i+1):
		kea[i]+=(l[j]*wa[j])**2
	for j in range(i+1):
		for k in range(j,N):
			if k!=j and k<=i:
				kea[i]+=2*l[j]*l[k]*wa[j]*wa[k]*np.cos(tha[j]-tha[k])
	kea[i]*=0.5*m[i]

Ea=kea+pea
ke=np.sum(kea,axis=0)
pe=np.sum(pea,axis=0)
E=np.sum(Ea,axis=0)
Emax=np.max(E)
E/=Emax
ke/=Emax
pe/=Emax

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.arrow(0,0,xa[0][frame],ya[0][frame],head_width=None,color='b')
	for i in range(1,N):
		plt.arrow(xa[i-1][frame],ya[i-1][frame],xa[i][frame]-xa[i-1][frame],ya[i][frame]-ya[i-1][frame],head_width=None,color='b')
	for i in range(N):
		circle=plt.Circle((xa[i][frame],ya[i][frame]),radius=0.05,fc='r')
		plt.gca().add_patch(circle)
	plt.title("N-Tuple Pendulum (N=%i)"%N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([lmin,lmax])
	plt.ylim([lmin,lmax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(tt[0:frame],ke[0:frame],'r',lw=0.5)
	plt.plot(tt[0:frame],pe[0:frame],'b',lw=0.5)
	plt.plot(tt[0:frame],E[0:frame],'g',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('ntuplependode.mp4', writer=writervideo)

plt.show()




