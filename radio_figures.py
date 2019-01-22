import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u

plt.rcParams['figure.autolayout']=False
plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['mathtext.rm']='Latin Modern Math'
plt.rcParams['font.size']=11
plt.rcParams['figure.figsize']=(10,10)

# scope (in pixel coordinates. The orginal fits images are 256 x 256)
scope_xmin = 60
scope_xmax = 210
scope_ymin = 60
scope_ymax = 210

# title label position (in pixel coordinates)
titlelabel = (65,201)

# beam
beam_maj = 3.5*u.mas/2
beam_min = 1.5*u.mas/2
beam_posangle = -5.97*u.deg

# define unrotated beam
phi = np.linspace(0.,2*np.pi)
beamx_unrot = np.cos(phi)*beam_min
beamy_unrot = np.sin(phi)*beam_maj

# define beam center coordinate (in world coordinates)
beamcenterx = Angle("13h09m48.0694s")
beamcentery = Angle("-23d22m53.401s")

# define rotated beam
beamx = beamcenterx + np.cos(beam_posangle.to('rad').value)*beamx_unrot + np.sin(beam_posangle.to('rad').value)*beamy_unrot
beamy = beamcentery + np.cos(beam_posangle.to('rad').value)*beamy_unrot - np.sin(beam_posangle.to('rad').value)*beamx_unrot

# re-center beam
beamx_unrot += beamcenterx
beamy_unrot += beamcentery

# my colormap (similar to a Difmap one)
my_cmap_colcodes = ['#000000','#03509C','#0085CC','#00AAAE','#48B26C','#89C540','#C3DB14','#FBDD02','#FEBC11','#F7931D','#ED1C24']
my_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',my_cmap_colcodes)

# files
datadir = './'
noise_img = 'NOISE.fits'
source_img = 'NOISE-EM170817.fits'
SJ_img = 'SJ_207d_5GHz-ZSOLT.fits'
C30_img = 'C30_207d_5GHz-ZSOLT.fits'
C45_img = 'C45_207d_5GHz-ZSOLT.fits'
#'NOISE-EM170817.fits'
#'SJ_207d_5GHz-ZSOLT.fits'
#'C30_207d_5GHz-ZSOLT.fits'

# Mooley's and our coordinates
pos_75d =     (Angle("13h09m48.068638s"),Angle("-23d22m53.3909s"))
pos_75d_err = (Angle("00h00m00.000008s"),Angle("00d00m00.0004s"))
pos_207d =    (Angle("13h09m48.0688006s"),Angle("-23d22m53.390765s"))
pos_207d_err =(Angle("00h00m00.00002s"),Angle("00d00m00.00025s"))
pos_230d =    (Angle("13h09m48.068831s"),Angle("-23d22m53.3907s"))
pos_230d_err =(Angle("00h00m00.000011s"),Angle("00d00m00.0004s"))

# open noise image and compute rms
f = fits.open(os.path.join(datadir,source_img))
hdu = f[0]
rms = np.std(hdu.data)

print("The noise image has an rms of {0:.2f} uJy".format(rms*1e6))

# open fits file
f = fits.open(os.path.join(datadir,source_img))

# get image hdu
hdu = f[0]

# re-reference position to J1312-2350 
"""
Om's note: the correction by Zsolt doesn't seem to work. 
In the email he sent, he wrote:

   48.0687231 + 0.0000775 ==>  RA 13 09 48,06880(06) - 0.000014s statistical error from the fit
   53.391670  + -0.000905 ==>  DEC -23 22 53.390(765)- 0.000245" statistical error from the fit
 
   This new position is within 458 microarcsec in RA, and 25 microarcsec in DEC of the Mooley+18 230d position.
   RA difference seems significant between 207d and 230d at the 2 sigma level, supporting the superluminal motion
   claimed by that group (or at least in agreement with their claim)!

But when I apply the corrections (7.75e-5 sec on the RA, -9.05e-4 arcsec on the DEC) I get a source image
which is totally off in comparison to Mooley's positions. So I worked out the corrections by matching the source 
image with Zsolt's position. 
"""
RA_correction_deg = 4.6e-4/3600. #arcseconds to degrees 
DEC_correction_deg = 8e-4/3600. #arcseconds to degrees

# add the correction to the central pixel position in the FITS header 
hdu.header['CRVAL1']+=RA_correction_deg
hdu.header['CRVAL2']+=DEC_correction_deg

# save the value, in order to set it in the next images
src_crval1 = hdu.header['CRVAL1']
src_crval2 = hdu.header['CRVAL2']

# construct wcs from fits header
w = wcs.WCS(hdu.header)

# print the position of the brightest pixel (to check that we did things correctly)
brightest = np.unravel_index(np.argmax(hdu.data[0,0]),hdu.data[0,0].shape) # find the pixel
brightest_RA,brightest_DEC = w.celestial.all_pix2world(brightest[0],brightest[1],0) # convert the pixel to world coordinates
brightest_RA = Angle((360.+brightest_RA)*u.deg) # make an angle object, which has methods to convert to hms
brightest_DEC = Angle(brightest_DEC*u.deg) # make an angle object, which has methods to convert to dms
print("Position of the birghtest pixel according to the wcs: ",brightest_RA.hms,brightest_DEC.dms)

# convert Mooley's and our positions to pixel coordinates
pix75d = np.array(w.celestial.all_world2pix(pos_75d[0].to('deg'),pos_75d[1],0))
pix75d_err = np.abs(np.array(w.celestial.all_world2pix((pos_75d[0]+pos_75d_err[0]).to('deg'),(pos_75d[1]+pos_75d_err[1]),0))-pix75d)
pix207d = np.array(w.celestial.all_world2pix(pos_207d[0].to('deg'),pos_207d[1],0))
pix207d_err = np.abs(np.array(w.celestial.all_world2pix((pos_207d[0]+pos_207d_err[0]).to('deg'),(pos_207d[1]+pos_207d_err[1]),0))-pix207d)
pix230d = np.array(w.celestial.all_world2pix(pos_230d[0].to('deg'),pos_230d[1],0))
pix230d_err = np.abs(np.array(w.celestial.all_world2pix((pos_230d[0]+pos_230d_err[0]).to('deg'),(pos_230d[1]+pos_230d_err[1]),0))-pix230d)

# find image frame coordinates
# RAsize,DECsize = hdu.data.shape[2:4]
# RAlims,DEClims = w.celestial.all_pix2world([0,RAsize],[0,DECsize],1)
# ra = np.linspace(RAlims[0],RAlims[1],RAsize)*u.deg
# dec = np.linspace(DEClims[0],DEClims[1],DECsize)*u.deg
# RA,DEC = np.meshgrid(ra,dec)

# plot image
fig = plt.figure()
ax = fig.add_subplot(221,projection=w,slices=('x','y',0,0))

plt.imshow(hdu.data[0,0]*1e6,cmap=my_cmap,vmin=-30,vmax=43.) # 1e6 converts to uJy
#plt.colorbar(label=r'brightness [$\mathrm{\mu}$Jy/beam]')
plt.contour(hdu.data[0,0]*1e6,levels=[-20,20,40],linestyles=['--','-','-','-','-'],colors='w',linewidths=0.8)

# beam
bx,by = w.celestial.all_world2pix(beamx.to('deg'),beamy.to('deg'),1)
plt.plot(bx,by,'-',c='w')

## bounding box
dx = 3
dy = 3
plt.plot([np.max(bx)+dx,np.max(bx)+dx,np.min(bx)-dx,np.min(bx)-dx,np.max(bx)+dx],
         [np.min(by)-dy,np.max(by)+dy,np.max(by)+dy,np.min(by)-dy,np.min(by)-dy],
         '-',color='w',lw=1)

# axes settings
ax.invert_yaxis()
ax.coords['RA'].set_major_formatter('hh:mm:ss.ssss')
ax.coords['RA'].display_minor_ticks(True)
ax.coords['DEC'].display_minor_ticks(True)
ax.coords['RA'].set_axislabel('')
ax.coords['RA'].set_ticklabel_visible(False)
ax.coords['RA'].set_ticks_position('t')
ax.coords['DEC'].set_ticks_position('l')

plt.ylabel('DEC',labelpad=-1)

# I would like to zoom a bit
ax.set_xlim([scope_xmin,scope_xmax])
ax.set_ylim([scope_ymin,scope_ymax])




# ---- create inset --------------------------------------------------------

## create axes (rectangle [left,bottom,width,heigth] in figure fraction units)
ax_inset = fig.add_axes([0.375,0.72,0.12,0.15],zorder=100)

# plot image and contours
ax_inset.imshow(hdu.data[0,0]*1e6,cmap=my_cmap,vmin=-30,vmax=43.)
ax_inset.contour(hdu.data[0,0]*1e6,levels=[-20,20,40],linestyles=['--','-','-','-','-'],colors='w',linewidths=0.8)

# define scope
ix_min = 105 # in pixels (the image is 256 x 256)
ix_max = 125
iy_min = 115
iy_max = 140

# define the ticks
xticks = [pix75d[0]-15,pix75d[0]-10,pix75d[0]-5,pix75d[0]-1e-10] # the 1e-10 is to make sure the result is 0 and not -0
yticks = [pix75d[1]-10,pix75d[1]-5,pix75d[1],pix75d[1]+5,pix75d[1]+10]

# what are the tick positions in the wcs?
dx_wcs = hdu.header['CDELT1']*3600.*1000. # pixel x coordinate increment in mas
dy_wcs = hdu.header['CDELT2']*3600.*1000. # pixel y coordinate increment in mas

# set the scope, ticks and labels
ax_inset.set_xlim([ix_min,ix_max])
ax_inset.set_ylim([iy_min,iy_max])
ax_inset.xaxis.set_ticks(xticks)
ax_inset.xaxis.set_ticklabels(["{0:.0f}".format((x-pix75d[0])*dx_wcs) for x in xticks])
ax_inset.yaxis.set_ticks(yticks)
ax_inset.yaxis.set_ticklabels(["{0:.0f}".format((y-pix75d[1])*dy_wcs) for y in yticks])
ax_inset.set_xlabel(r'$\Delta$RA [mas]')
ax_inset.set_ylabel(r'$\Delta$DEC [mas]')

# plot Mooley's positions
ax_inset.errorbar([pix75d[0],pix230d[0]],[pix75d[1],pix230d[1]],xerr=(pix75d_err[0],pix230d_err[0]),yerr=(pix75d_err[1],pix230d_err[1]),marker='o',color='k',capsize=3,ls='None',markersize=3)
#ax_inset.errorbar([pix207d[0]],[pix207d[1]],xerr=(pix207d_err[0]),yerr=(pix207d_err[1]),marker='o',color='k',capsize=3,ls='None',markersize=3) # this is our position

# --------------- go back to bigger image -----------------------------------

# annotation (image title)
plt.sca(ax) # go back to the bigger image axes (sca = Select Current Axes)
plt.annotate(xy=titlelabel,s='Real source image')

#plt.savefig("radiomap_real.pdf")

# --------------------------- SJ image ----------------------------------------

# open fits file
f = fits.open(os.path.join(datadir,SJ_img))

# get image hdu
hdu = f[0]

# re-reference position to match source image
hdu.header['CRVAL1']=src_crval1
hdu.header['CRVAL2']=src_crval2

# construct wcs
w = wcs.WCS(hdu.header)

# plot image

ax = fig.add_subplot(222,projection=w) # create axes
plt.imshow(hdu.data*1e6,cmap=my_cmap,vmin=-30,vmax=43.)
plt.contour(hdu.data*1e6,levels=[-20,20,40],linestyles=['--','-','-','-','-'],colors='w',linewidths=0.8)

# beam
bx,by = w.celestial.all_world2pix(beamx_unrot.to('deg'),beamy_unrot.to('deg'),1)
plt.plot(bx,by,'-',c='w')

## beam bounding box
dx = 3 # distance from beam border
dy = 3
plt.plot([np.max(bx)+dx,np.max(bx)+dx,np.min(bx)-dx,np.min(bx)-dx,np.max(bx)+dx],
         [np.min(by)-dy,np.max(by)+dy,np.max(by)+dy,np.min(by)-dy,np.min(by)-dy],
         '-',color='w',lw=1)


# axes settings
ax.invert_yaxis()
ax.coords['RA'].set_major_formatter('hh:mm:ss.ssss')
ax.coords['RA'].display_minor_ticks(True)
ax.coords['DEC'].display_minor_ticks(True)

ax.coords['DEC'].set_axislabel('')
ax.coords['DEC'].set_ticklabel_visible(False)

ax.coords['RA'].set_axislabel('')
ax.coords['RA'].set_ticklabel_visible(False)

ax.coords['RA'].set_ticks_position('t')
ax.coords['DEC'].set_ticks_position('r')


# annotation
plt.annotate(xy=titlelabel,s='Successful jet \nsimulated image + real noise',va='top')

# I would like to zoom a bit
ax.set_xlim([scope_xmin,scope_xmax])
ax.set_ylim([scope_ymin,scope_ymax])


#plt.savefig("radiomap_SJ.pdf")

# ----------------------------------------- CJ30 ----------------------------------------

# open fits file
f = fits.open(os.path.join(datadir,C30_img))

# get image hdu
hdu = f[0]

print("C30 max: ",np.max(hdu.data)*1e6)

# re-reference position to match source image
RA_correction_deg = 7.75e-5/3600. #arcseconds to degrees
DEC_correction_deg = 9.05e-4/3600. #arcseconds to degrees

hdu.header['CRVAL1']=src_crval1
hdu.header['CRVAL2']=src_crval2

# construct wcs
w = wcs.WCS(hdu.header)

# plot image
#fig = plt.figure()
ax = fig.add_subplot(223,projection=w)
plt.imshow(hdu.data*1e6,cmap=my_cmap,vmin=-30,vmax=43.)
#plt.colorbar(label=r'flux density [$\mathrm{\mu}$Jy/beam]')
plt.contour(hdu.data*1e6,levels=[-20,20,40],linestyles=['--','-','-','-','-'],colors='w',linewidths=0.8)

# beam
bx,by = w.celestial.all_world2pix(beamx_unrot.to('deg'),beamy_unrot.to('deg'),1)
plt.plot(bx,by,'-',c='w')

## bounding box
dx = 3
dy = 3
plt.plot([np.max(bx)+dx,np.max(bx)+dx,np.min(bx)-dx,np.min(bx)-dx,np.max(bx)+dx],
         [np.min(by)-dy,np.max(by)+dy,np.max(by)+dy,np.min(by)-dy,np.min(by)-dy],
         '-',color='w',lw=1)


# axes settings
ax.invert_yaxis()
ax.coords['RA'].set_major_formatter('hh:mm:ss.ssss')
ax.coords['RA'].display_minor_ticks(True)
ax.coords['DEC'].display_minor_ticks(True)

ax.coords['RA'].set_ticks_position('b')
ax.coords['DEC'].set_ticks_position('l')

plt.xlabel('RA')
plt.ylabel('DEC',labelpad=-1)

# annotation
plt.annotate(xy=titlelabel,s='Choked jet cocoon (' + r'$\theta_\mathrm{c}=30^\circ$' + ') \nsimulated image + real noise',va='top')

# Zoom a bit
ax.set_xlim([scope_xmin,scope_xmax])
ax.set_ylim([scope_ymin,scope_ymax])

#plt.savefig("radiomap_C30.pdf")

# CJ45 ----------------------------------------

# open fits file
f = fits.open(os.path.join(datadir,C45_img))

# get image hdu
hdu = f[0]

# re-reference position to match source image
RA_correction_deg = 7.75e-5/3600. #arcseconds to degrees
DEC_correction_deg = 9.05e-4/3600. #arcseconds to degrees

hdu.header['CRVAL1']=src_crval1
hdu.header['CRVAL2']=src_crval2

# construct wcs
w = wcs.WCS(hdu.header)

# plot image
#fig = plt.figure()
ax = fig.add_subplot(224,projection=w)
ims = plt.imshow(hdu.data*1e6,cmap=my_cmap,vmin=-30,vmax=43.)
#plt.colorbar(label=r'flux density [$\mathrm{\mu}$Jy/beam]')
plt.contour(hdu.data*1e6,levels=[-20,20,40],linestyles=['--','-','-','-','-'],colors='w',linewidths=0.8)

# beam
bx,by = w.celestial.all_world2pix(beamx_unrot.to('deg'),beamy_unrot.to('deg'),1)
plt.plot(bx,by,'-',c='w')

## bounding box
dx = 3
dy = 3
plt.plot([np.max(bx)+dx,np.max(bx)+dx,np.min(bx)-dx,np.min(bx)-dx,np.max(bx)+dx],
         [np.min(by)-dy,np.max(by)+dy,np.max(by)+dy,np.min(by)-dy,np.min(by)-dy],
         '-',color='w',lw=1)

# axes settings
ax.invert_yaxis()
ax.coords['RA'].set_major_formatter('hh:mm:ss.ssss')
ax.coords['RA'].display_minor_ticks(True)
ax.coords['DEC'].display_minor_ticks(True)

ax.coords['DEC'].set_axislabel('')
ax.coords['DEC'].set_ticklabel_visible(False)

ax.coords['RA'].set_ticks_position('b')
ax.coords['DEC'].set_ticks_position('r')


plt.xlabel('RA')


# annotation
plt.annotate(xy=titlelabel,s='Choked jet cocoon (' + r'$\theta_\mathrm{c}=45^\circ$' + ') \nsimulated image + real noise',va='top')

ax.set_xlim([scope_xmin,scope_xmax])
ax.set_ylim([scope_ymin,scope_ymax])

# reduce space between axes
plt.subplots_adjust(hspace=0.018,wspace=0.018)

# ----------------------- colorbar
cax = fig.add_axes([0.92,0.1,0.025,0.8])
plt.colorbar(ims,cax=cax,label=r'flux density [$\mathrm{\mu}$Jy/beam]')


# save figure!
plt.savefig("radiomaps_all.pdf")


plt.show()
