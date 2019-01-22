import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs 
from scipy.optimize import minimize
from scipy.signal import fftconvolve as convolve2d
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import threading

recompute = False

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['font.size']=12
plt.rcParams['figure.figsize']=(5.,4.)
plt.rcParams['figure.autolayout']=True

our_beam_maj = 3.5 #mas
our_beam_min = 1.5 #mas
our_beam_posangle = -6. #deg

totflux = 47.
totflux_err = 9.

our_pf = 42.

sizes_x = np.arange(0.0,5.01,0.1)
sizes_y = np.arange(0.0,10.01,0.1)

def g2d(x,y,mx,my,sx,sy,r):
    
    t = np.arctan2(y,x)
    R = (x**2+y**2)**0.5
    
    tm = np.arctan2(my,mx)
    Rm = (mx**2+my**2)**0.5
    
    X = R*np.cos(t-r/180*np.pi)
    Y = R*np.sin(t-r/180*np.pi)
    
    MX = Rm*np.cos(tm-r/180*np.pi)
    MY = Rm*np.sin(tm-r/180*np.pi)
    
    return np.exp(-0.5*(((X-MX)/sx)**2+((Y-MY)/sy)**2))

def convolve_with_beam(img,beam_img):
    img_conv = convolve2d(img,beam_img)
    return img_conv

def create_source_image(size,xres,yres):
    sigma = size/2.355
    x,y = np.meshgrid(np.arange(-4*sigma,4*sigma,xres),np.arange(-4*sigma,4*sigma,yres))
    source_img = g2d(x,y,0,0,sigma,sigma,0.)
    
    return source_img,x,y

def create_source_image_elliptical(size_x,size_y,xres,yres,posangle=0.):
    sigma = max(size_x,size_y)/2.355
    x,y = np.meshgrid(np.arange(-3*sigma,3*sigma,xres),np.arange(-3*sigma,3*sigma,yres))
    source_img = g2d(x,y,0,0,size_x/2.355,size_y/2.355,posangle)
    
    return source_img,x,y

class compute_like (threading.Thread) :

    def __init__(self, threadID, i, j):

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.i = i
        self.j = j
        
    def run(self):
        print("(id: {2:d}) computing likelihood for source model with sizes ({0:.3g}, {1:.3g}) mas ".format(sizes_x[self.i],sizes_y[self.j],self.threadID))
        source_img,source_x,source_y = create_source_image_elliptical(sizes_x[self.i],sizes_y[self.j],map_res,map_res)
        
        pix_area = (source_x[1,1]-source_x[0,0])*(source_y[1,1]-source_y[0,0])
        
        source_img*=totflux/np.sum(pix_area*source_img.ravel())
        
        source_conv = convolve_with_beam(source_img,beam_img)*pix_area
        sc_x,sc_y = np.meshgrid(np.arange(0.,source_conv.shape[1]*map_res,map_res),np.arange(0.,source_conv.shape[0]*map_res,map_res)) # axes coordinates in mas
        
        sc_max = np.unravel_index(np.argmax(source_conv.ravel()),source_conv.shape)
        
        px,py = source_conv.shape
        
        n_iter = 1000
        pfs = np.empty([n_iter,1000])
        i0s = np.random.randint(0,noise_img.shape[0]-px,n_iter)
        j0s = np.random.randint(0,noise_img.shape[1]-py,n_iter)
        
        M = np.random.normal(totflux,totflux_err,1000)/totflux*source_conv[sc_max]
        N = noise_img[i0s,j0s]
        pfs = (N.reshape([n_iter,1]) + M).ravel()
        
        kde = gaussian_kde(pfs)
        pflike[self.i,self.j] = kde(our_pf)


if recompute:
    
    map_res = 0.05 

    f = fits.open('ourbeam.fits')
    hdu = f[0]
    beam_img = hdu.data[0,0]
    lx,ly = beam_img.shape
    wx,wy = 476,204
    beam_img = beam_img[(lx-wx)//2:(lx+wx)//2,(ly-wy)//2:(ly+wy)//2]
        
    f = fits.open('noise_hires.fits')
    
    hdu = f[0]
    noise_img = hdu.data[0,0]
    noise_img*=1e6
    
    pflike = np.zeros([len(sizes_x),len(sizes_y)])
        
    thread_list = []
    k = 0
    for i in range(1,len(sizes_x)):
        for j in range(1,len(sizes_y)):
            thread_list.append(compute_like(k,i,j))
            k = k+1
    
    n_cpus = 4
    
    while len(thread_list)>0:
        
        active_threads = []
    
        for n in range(n_cpus):
            if len(thread_list)>0:
                active_threads.append(thread_list.pop())
        
        for t in active_threads:
            t.start()
        
        for t in active_threads:
            t.join()
        
    pflike[0,:] = pflike[1,:]
    pflike[:,0] = pflike[:,1]
    
    np.save("likelihood_2d.npy",pflike)

else:
    pflike = np.load("likelihood_2d.npy")
    
pflike = gaussian_filter(pflike,0.9)

size_prior = np.ones([len(sizes_x),len(sizes_y)]) 

posterior = pflike*size_prior 

posterior /= np.sum(posterior.ravel()*(sizes_x[1]-sizes_x[0])*(sizes_y[1]-sizes_y[0]))

rav_posterior = posterior.ravel()
rav_idx = np.arange(len(rav_posterior))
sorted_idx = np.argsort(rav_posterior)
rav_cum_posterior = np.zeros(len(rav_posterior))
rav_cum_posterior[sorted_idx] = np.cumsum(rav_posterior[sorted_idx])
cum_posterior = np.zeros(posterior.shape)
cum_posterior[np.unravel_index(rav_idx,posterior.shape)]=rav_cum_posterior*(sizes_x[1]-sizes_x[0])*(sizes_y[1]-sizes_y[0])

# plot the contours
plt.contour(sizes_x,sizes_y,cum_posterior.T,levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

plt.xlabel('size along beam minor axis [mas]')
plt.ylabel('size along beam major axis [mas]')
plt.show()
