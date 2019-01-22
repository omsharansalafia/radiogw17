import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=8,4
plt.rcParams['figure.autolayout']=True

# analytic structure functional forms
def Eth_PL(th,thc,Ec,s):
    return Ec/(1+(th/thc)**s)

def Gth_PL(th,thc,Gc,s):
    return 1. + (Gc-1.)/(1+(th/thc)**s)

def Eth_PL(th,thc,Ec,s):
    return Ec/(1+(th/thc)**s)

def Eth_G(th,thc,Ec,tw):
    e = Ec*np.exp(-th**2/2./thc**2)
    e[th>tw]=0.
    return e

# load mcmc chain
chain0 = np.load("AT2017gfo_eefix_chain.npz")['arr_0']
logchisq0 = np.load('AT2017gfo_eefix_lnprob.npy')    

burnin = 1000

chain = chain0[(logchisq0>0.)][burnin::10,:]

trnsf = [lambda x:x/np.pi*180, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x/np.pi*180]

logE0 = trnsf[3](chain[:,3])
s1 = trnsf[4](chain[:,4])
logG0 = trnsf[5](chain[:,5])
s2 = trnsf[6](chain[:,6])
th0 = trnsf[7](chain[:,7])

th = np.logspace(-1,2,100)
E = np.empty([1000,100])
G = np.empty([1000,100])

for j,i in enumerate(np.random.randint(0,len(th0),1000)):
    E[j] = Eth_PL(th,th0[i],10**logE0[i],s1[i])/(4*np.pi)
    G[j] = Gth_PL(th,th0[i],10**logG0[i],s2[i])

Em = np.zeros(100)
EM = np.zeros(100)
Gm = np.zeros(100)
GM = np.zeros(100)

for j in range(100):
    Em[j] = np.percentile(E[:,j],16.)
    EM[j] = np.percentile(E[:,j],84.)
    Gm[j] = np.percentile(G[:,j],16.)
    GM[j] = np.percentile(G[:,j],84.)

# parametrized structures
DAv18 = {'thc': 2.0,
         'Ec': 1e52,
         's1': 3.5,
         'Gc': 110.,
         's2': 2.,
         'type':'PL'}

gg18 = {'thc': 3.4,
         'Ec': 10**52.4,
         's1': 5.5,
         'Gc': 10**2.4,
         's2': 3.5,
         'type':'PL'}

Tr18 = {'thc': 0.057/np.pi*180,
         'Ec': 10**52.73,
         'thw': 0.62/np.pi*180,
         'type':'G'}


# make plots
fig, (ax1,ax2) = plt.subplots(1,2)

# ------------- dE/dOmega ---------------------------------------------
plt.sca(ax1)
plt.xlabel(r'$\theta$ [deg]')
plt.ylabel(r'$dE/d\Omega$ [erg/sr]')

plt.tick_params(axis='y',which='both',left=True,right=True,labelleft=True,labelright=False)

plt.fill_between(th,Em,EM,color='pink',edgecolor='None',zorder=-1)
plt.plot(th,Eth_PL(th,gg18['thc'],gg18['Ec'],gg18['s1'])/(4*np.pi),'-',color='red',lw=3,label='This work')

plt.loglog()
plt.xlim([5e-1,90])
plt.ylim([1e47,1e53])

plt.legend(loc='lower left',frameon=False)


# ------------- Gamma --------------------------------------------------
plt.sca(ax2)
plt.xlabel(r'$\theta$ [deg]')
plt.ylabel(r'$\Gamma-1$')

plt.tick_params(axis='y',which='both',left=True,right=True,labelleft=False,labelright=True)
ax2.yaxis.set_label_position('right')

plt.fill_between(th,Gm-1.,GM-1.,color='pink',edgecolor='None',zorder=-1)
plt.plot(th,Gth_PL(th,gg18['thc'],gg18['Gc'],gg18['s2'])-1.,'-',color='red',lw=3)


plt.loglog()
plt.xlim([5e-1,90])
plt.ylim([1e-1,1e4])

plt.subplots_adjust(wspace=0.1)

plt.show()




