import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas
from matplotlib.ticker import MultipleLocator,LinearLocator
from sklearn.decomposition import PCA
from scipy.stats.kde import gaussian_kde

c = 3e10

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['font.size']=10
plt.rcParams['figure.figsize']=(8.,8.)
plt.rcParams['figure.autolayout']=False

# load chain
chain0 = np.load("AT2017gfo_eefix_chain.npz")['arr_0']
logchisq0 = np.load("AT2017gfo_eefix_lnprob.npy")

burnin = 1000

chain = chain0[(logchisq0>0.)][burnin:,:]

parnames = [r'$\theta_\mathrm{v}/\mathrm{deg}$',r'$\log(n/\mathrm{cm^{-3}})$',r'$\log(\epsilon_\mathrm{B})$',r'$\log(E_\mathrm{c}/\mathrm{erg})$',r'$s_1$',r'$\log(\Gamma_\mathrm{c})$',r'$s_2$',r'$\theta_\mathrm{c}/\mathrm{deg}$']
trnsf = [lambda x:x/np.pi*180, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x, lambda x:x/np.pi*180]

# plot parameter chains
tv = trnsf[0](chain[:,0])
logn = trnsf[1](chain[:,1])
logeb = trnsf[2](chain[:,2])
logE0 = trnsf[3](chain[:,3])
s1 = trnsf[4](chain[:,4])
logG0 = trnsf[5](chain[:,5])
s2 = trnsf[6](chain[:,6])
th0 = trnsf[7](chain[:,7])

pars = [tv,logn,logeb,logE0,s1,logG0,s2,th0]
    
bestfit = np.empty([len(chain[0,:]),3])

Ntot = len(tv)
ii = np.random.randint(0,Ntot-1,100)

pca = PCA()
pca.fit(chain)
xt = pca.transform(chain)
bestfit[:,0] = pca.inverse_transform([corner.quantile(x,[0.5])[0] for x in xt.T])

tvb = trnsf[0](bestfit[0,0])
lognb = trnsf[1](bestfit[1,0])
logebb = trnsf[2](bestfit[2,0])
logE0b = trnsf[3](bestfit[3,0])
s1b = trnsf[4](bestfit[4,0])
logG0b = trnsf[5](bestfit[5,0])
s2b = trnsf[6](bestfit[6,0])
th0b = trnsf[7](bestfit[7,0])

bests = [tvb,lognb,logebb,logE0b,s1b,logG0b,s2b,th0b]

fig = corner.corner(np.array(pars).T, labels=parnames, smooth=1.5/2.355,
                    truths=bests, truth_color='red', color='black', 
                    quantiles=[0.16,0.84], levels=[1-np.exp(-0.5),1-np.exp(-2.)], 
                    show_titles=False, label_kwargs = {'size':12}, 
                    title_kwargs = {'size':12, 'family':'Liberation Serif'})

plt.show()
