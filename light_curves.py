import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,LinearLocator
import matplotlib
import pandas
from astropy.cosmology import Planck15 as cosmo
from astropy.time import Time

recompute = False

if recompute:
    import surface


plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['font.size']=12
plt.rcParams['figure.figsize']=(6.,4.)
plt.rcParams['figure.autolayout']=True

# gw time
tgw = Time("2017-08-17 14:00:00")

# VLBI time
tVLBI = Time("2018-03-13 00:00:00")
tVLBI = (tVLBI-tgw).to('d').value

# e-EVN times
tEVN1 = Time("2018-03-27 00:00:00")
tEVN1 = (tEVN1-tgw).to('d').value
tEVN2 = Time("2018-05-26 00:00:00")
tEVN2 = (tEVN2-tgw).to('d').value

c = 3e10

# GW170817
dL = 40.*3.08e24 #cm
z = 0.01

xlim = [5,4e2]
ylim = [1e-3,2e-1]


# ----- parametri plot -------

# multiplicative factors
xmult = {'radio0.6':1.5**3,'radio1.4':1.5**2, 'radio3':1.5, 'radio6':1, 'opt':400, 'X-ray':5000}

# frequenze
#nu_obs = {'radio0.6':6.1e8,'radio1.4':1.4e9, 'radio3': 3e9, 'radio6':6e9, 'opt': 5e14, 'X-ray':2.42e17} #Hz
nu_obs = {'radio3': 3e9, 'opt': 5e14, 'X-ray':2.42e17} #Hz

# colori
colors = {'radio0.6':'grey','radio1.4':'maroon', 'radio3': 'red', 'radio6': 'orange', 'opt': '#3FE90D', 'X-ray':'blue'}
#colors = {'radio0.6':'grey','radio1.4':'maroon', 'radio3': '#FF2E00', 'radio6': 'orange', 'opt': '#CB00FF', 'X-ray':'blue'}
colors2 = {'radio0.6':'#803333','radio1.4':'#916191', 'radio3': '#FF8080', 'radio6': '#FFD280', 'opt': '#61C261', 'X-ray':'#8080FF'}

# marker
markers = {'radio0.6':'o','radio1.4':'s', 'radio3': '*', 'radio6': 'h', 'opt': '*', 'X-ray':'*'}

# legenda
legend_labels = {'radio0.6': r'$\,610\,\mathrm{MHz}$' + r' ($\times 27/8$)','radio1.4': r'$\,1.4\,\mathrm{GHz}$' + r' ($\times 9/4$)', 'radio3': r'$\,3\,\mathrm{GHz}$' + r' ($\times 3/2$)', 'radio6': r'$\,6\,\mathrm{GHz}$', 'opt': r'$5\times 10^{14}\,\mathrm{Hz}$ ' + r'($\times {0:.0f}$)'.format(xmult['opt']), 'X-ray': r'$\,1\,\mathrm{keV}$' + r' ($\times {0:.0f}$)'.format(xmult['X-ray'])}

# ------------------------


# ------------- plot -------------

fig = plt.figure()
ax2 = fig.add_axes([0.105,0.15,0.8,0.8])
ax22 = ax2.twinx()


# ---------------- plot models    

if True: # set to False if you do not want to plot this model
    
    Mooley_E0 = 1.5e52
    Mooley_bmin = 0.89
    Mooley_n = 1.8e-4
    Mooley_Gmax = 6.
    eB = 0.01
    ee = 0.05
    Mooley_p = 2.14
    Mooley_alpha = 6.
    theta_view = 30.
    theta_jet = 45.
    
    if recompute:
        t,F = surface.surface.velprofile_lightcurve([nu_obs[nu] for nu in nu_obs.keys()],z,dL,Mooley_n,Mooley_E0,Mooley_alpha,Mooley_bmin,Mooley_Gmax,theta_jet/180.*np.pi,theta_view/180.*np.pi,ee,eB,Mooley_p, th_res = 50, phi_res = 30, lc_res=100,t0=3,t1=400)
        np.save('cocoon_lcs_t.npy',t)
        np.save('cocoon_lcs_F.npy',F)
    else:
        t = np.load('cocoon_lcs_t.npy')
        F = np.load('cocoon_lcs_F.npy')
    
    
    for i,nu in enumerate(nu_obs.keys()):   
        F[:,i] = (F[:,i]*xmult[nu]) # multiplicative factors
        ax2.plot(t,F[:,i],'-.',color=colors[nu],label='',lw=3)
    
# Structured Jet
if True:

    tv = 15./180*np.pi
    n = 10**(-3.6)
    ee = 0.095
    eB = 10**(-3.9)
    p=2.14
    
    E0 = 10**(52.5)
    s1 = 5.5
    G0 = 10**(2.4)
    s2   = 3.5
    th0 = 3.4/180*np.pi
    
    theta = np.logspace(-4,1.0,300)
    Eth = E0/(1.+(theta/th0)**s1)
    Gth = 1. + (G0-1.)/(1.+(theta/th0)**s2)
    
    if recompute:
        t,F = surface.structured_lightcurve([nu_obs[nu] for nu in nu_obs.keys()],z,dL,n,theta,Eth,Gth,tv,ee,eB,p, thmax=1., th_res = 50, phi_res = 30, lc_res=100, t0=xlim[0],t1=xlim[1])
        np.save('jet_lcs_t.npy',t)
        np.save('jet_lcs_F.npy',F)
    else:
        t = np.load('jet_lcs_t.npy')
        F = np.load('jet_lcs_F.npy')

    # plot

    for i,nu in enumerate(nu_obs.keys()):    
        F[:,i] = (F[:,i]*xmult[nu]) # multiplicative factors
        ax2.plot(t,F[:,i],'-',color=colors[nu],label='',lw=3)
        

ax2.loglog()
ax22.loglog()

plt.show()
