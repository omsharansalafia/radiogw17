import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from astropy.time import Time

recompute = False

if recompute:
    import surface

# unit conversions
cgs_to_uJy = 1e29 # uJy/(erg cm^-2 s^-1)
cgs_to_mJy = 1e26 # mJy/(erg cm^-2 s^-1)
sr_to_mas2 = 4.25e16 # mas^2/sr
rad_to_mas = 206265000. # mas/rad

# extrinsic information about GW170817
dL = 41*3.08e24 #cm
z = 0.008
tgw = Time("2017-08-17 12:41:04")

# our g-VLBI observation time and frequency
t_obs = 207.4 #(Time("2018-03-12 23:59:59") - tgw).to('d').value
nu_obs = 5e9

print("Predicted images {0:.2f} days after merger, at {1:.1f} GHz".format(t_obs,nu_obs/1e9))

# structured jet image

## resolution
xres = 10
yres = 20

# jet parameters
tv = 15.5/180*np.pi
n = 10**(-3.7154)
ee = 0.1
eB = 10**(-3.56)
p=2.15

E0 = 10**(52.26)
s1 = 5.45
G0 = 10**(2.5)
s2   = 3.9
th0 = 3.54/180*np.pi

## compute structure
theta = np.logspace(-4,0.5,100)
E0th = E0/(1.+(theta/th0)**s1)
G0th = 1. + (G0-1.)/(1.+(theta/th0)**s2)

Rth = np.zeros([len(theta),1000])
Gth = np.zeros([len(theta),1000])

## make image
if recompute:
    for i in range(len(theta)):
        Rth[i],Gth[i],tj = surface.surface.dynamics_noSE(E0th[i],G0th[i],n,np.max(theta))
    imgJ,XJ,YJ = surface.image.image_interior_structured(t_obs*(1.+z),nu_obs*(1.+z),tv,theta,Rth,Gth,n,ee,eB,p,phires=30,xres=xres,yres=yres,zres=1000) # jet
    imgC,XC,YC = surface.image.image_interior_structured(t_obs*(1.+z),nu_obs*(1.+z),np.pi+tv,theta,Rth,Gth,n,ee,eB,p,phires=30,xres=xres,yres=yres,zres=1000) # counterjet
    np.save('jetimg.npy',(imgJ,XJ,YJ))
    np.save('jetcimg.npy',(imgC,XC,YC))
else:
    imgJ,XJ,YJ = np.load('jetimg.npy')
    imgC,XC,YC = np.load('jetcimg.npy')

# sum the jet and counterjet images
img = np.zeros([xres,2*yres])
x = np.linspace(min(np.min(XJ),np.min(XC)),max(np.max(XJ),np.max(XC)),xres)
y = np.linspace(min(np.min(YJ),np.min(YC)),max(np.max(YJ),np.max(YC)),2*yres)
X,Y = np.meshgrid(x,y)
X = X.T
Y = Y.T

XJmin = np.min(XJ)
XJmax = np.max(XJ)
YJmin = np.min(YJ)
YJmax = np.max(YJ)

XCmin = np.min(XC)
XCmax = np.max(XC)
YCmin = np.min(YC)
YCmax = np.max(YC)


for i in range(xres):
    for j in range(2*yres):
        img[i,j] += map_coordinates(imgJ,[[(X[i,j]-XJmin)/(XJmax-XJmin)*xres],[(Y[i,j]-YJmin)/(YJmax-YJmin)*yres]],order=1)
        img[i,j] += map_coordinates(imgC,[[(X[i,j]-XCmin)/(XCmax-XCmin)*xres],[(Y[i,j]-YCmin)/(YCmax-YCmin)*yres]],order=1)

plt.pcolormesh(X,Y,img)
plt.show()

# isotropic outflow image

## resolution
xres = 10
yres = 10

## parameters
E0 = 1.5e52
bmin = 0.89
n = 1.8e-4
Gmax = 4
eB = 0.01
ee = 0.1
p = 2.14
alpha = 6.
theta_view = 30
theta_jet = 60

## make image with interior
if recompute:
    ## compute dynamics
    R,G = surface.dynamics.velocity_profile_deceleration(E0,alpha,bmin,Gmax,n)

    imgJ,XJ,YJ = surface.image.image_interior_tophat(t_obs*(1.+z),nu_obs*(1.+z),theta_view/180*np.pi,theta_jet/180*np.pi,R,G,n,ee,eB,p,thres=300,phires=100,xres=xres,yres=yres,zres=1000) # cocoon
    imgC,XC,YC = surface.image.image_interior_tophat(t_obs*(1.+z),nu_obs*(1.+z),np.pi+theta_view/180*np.pi,theta_jet/180*np.pi,R,G,n,ee,eB,p,thres=300,phires=100,xres=xres,yres=yres,zres=1000) # counter cocoon
    np.save('cimg.npy',(imgJ,XJ,YJ))
    np.save('ccimg.npy',(imgC,XC,YC))
else:
    imgJ,XJ,YJ = np.load('cimg.npy')
    imgC,XC,YC = np.load('ccimg.npy')


# sum the cocoon and counter-cocoon images
img = np.zeros([xres,2*yres])
x = np.linspace(min(np.min(XJ),np.min(XC)),max(np.max(XJ),np.max(XC)),xres)
y = np.linspace(min(np.min(YJ),np.min(YC)),max(np.max(YJ),np.max(YC)),2*yres)
X,Y = np.meshgrid(x,y)
X = X.T
Y = Y.T

XJmin = np.min(XJ)
XJmax = np.max(XJ)
YJmin = np.min(YJ)
YJmax = np.max(YJ)

XCmin = np.min(XC)
XCmax = np.max(XC)
YCmin = np.min(YC)
YCmax = np.max(YC)


for i in range(xres):
    for j in range(2*yres):
        img[i,j] += map_coordinates(imgJ,[[(X[i,j]-XJmin)/(XJmax-XJmin)*xres],[(Y[i,j]-YJmin)/(YJmax-YJmin)*yres]],order=1)
        img[i,j] += map_coordinates(imgC,[[(X[i,j]-XCmin)/(XCmax-XCmin)*xres],[(Y[i,j]-YCmin)/(YCmax-YCmin)*yres]],order=1)

plt.pcolormesh(X,Y,img)
plt.show()

