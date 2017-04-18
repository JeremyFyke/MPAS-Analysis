"""
Computation and plotting of model/observation ocean transects.

Authors
-------
Jeremy Fyke

Last Modified
-------------
04/14/2017
"""

import numpy as np
import scipy.io
import math
from netCDF4 import Dataset
from netCDF4 import MFDataset
from pyproj import Geod
from transect_generator import transectGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker

SOSEdataFile="/lustre/scratch2/turquoise/jer/THETA_AnnualAvg.mat"
vnames=['THETA_2005','THETA_2006','THETA_2007','THETA_2008','THETA_2009','THETA_2010']
vnames=['THETA_2010']
SOSEgridFile="/lustre/scratch2/turquoise/jer/SOSEgrid.mat"
MPAS_yrs=2
MPAS_yre=2
LoadSOSEData=1

d2r=math.radians(1.)
pi=math.pi

# load SOSE data
if LoadSOSEData==1:
    print("Loading SOSE data...")
    nSOSECells=2160*320
    nSOSEDepth=42
    nSOSEYears=len(vnames)
    ## load, time-average, and massage into form for input into transectGenerator
    SOSETemperature=np.zeros((nSOSECells,nSOSEDepth))
    for n,f in enumerate(vnames):
	print("Loading time slice: "+f)
    arr=scipy.io.loadmat(SOSEdataFile)[f]
    SOSETemperature[:,:]=SOSETemperature[:,:]+arr.reshape(-1,arr.shape[-1])
    SOSETemperature=SOSETemperature[:,:]/nSOSEYears
    SOSElon=np.ravel(scipy.io.loadmat(SOSEgridFile)['XC'])*d2r #degrees, 0->2pi
    SOSElat=np.ravel(scipy.io.loadmat(SOSEgridFile)['YC'])*d2r #degrees, -pi/2->pi/2
    SOSEdepth=       scipy.io.loadmat(SOSEgridFile)['RC'] #m, 0->-5575
    SOSEdepth=np.tile(np.transpose(SOSEdepth),(nSOSECells,1))
    for x in range(nSOSECells):
	if SOSETemperature[x,0]==0.0:
            SOSETemperature[x,0]=np.nan #nan land cells, in the surface layer..
    z=np.any(np.isnan(SOSETemperature),axis=1) #... get land cell horizontal indices
    SOSETemperature=SOSETemperature[-z,:]#... and remove these cells from SOSE
    SOSEdepth=SOSEdepth[-z,:]
    SOSElat=SOSElat[-z]
    SOSElon=SOSElon[-z]
    nSOSECells=np.size(SOSElon) #revise # of SOSE cells to just ocean
    # finally, set rest of non-ocean undersea points to zero
    i=np.where(SOSETemperature==0.0)
    SOSETemperature[i]=np.nan

MPASgridFile="/lustre/scratch2/turquoise/jer/MPASO_files_from_Edison/MPAS_grid.nc"
f=Dataset(MPASgridFile)
MPASlat=f.variables["latCell"][:] #-pi/2->pi/2
MPASlon=f.variables["lonCell"][:] #0->2pi

# define transects here.  This could be done elsewhere.
print("Making transects...")
g=Geod(ellps='sphere')
nTransectResolution=20000. #m
nTransectDepths=300
transectDistance=list()
transectLonLats=list()
ntransectPoints=list()
endPoints=np.zeros(4)
for lon in np.arange(-175,175,5): #start lat, start lon, end lat, end lon.
#for lon in [0]: #start lat, start lon, end lat, end lon.  ***Must be monotonically increasing or constant lat & lon values
    endPoints[0]=-80.
    endPoints[1]=lon
    endPoints[2]=-60.
    endPoints[3]=lon
    endPoints[:]=endPoints[:]*d2r

    # for code logic, start lon/lat must be .le. end lon/lat
    if (endPoints[0]>endPoints[2]) | (endPoints[1]>endPoints[3]):
       raise ValueError("Error: start lon/lat .gt. end lon/lat.")

    #Construct transect points in a great circle from start to end lat/lon points
    d=g.inv(endPoints[1],endPoints[0],
            endPoints[3],endPoints[2],radians=True)
    nt=np.floor(d[2]/nTransectResolution).astype(int)
    transectPoints=g.npts(endPoints[1],endPoints[0],
                          endPoints[3],endPoints[2],nt,radians=True)

    # append transect info to lists of transect info
    ntransectPoints.append(nt)
    transectDistance.append(d[2])
    transectLonLats.append(transectPoints)

# get final count of transects to process
nTransects=len(transectDistance)

#Loop over MPAS years, making set of transects for each year
for yr in np.arange(MPAS_yrs,MPAS_yrs+1):
    yrlong="%04d"%yr
    print("Generating transects for "+yrlong)
    MPASdataFile="/lustre/scratch2/turquoise/jer/MPASO_files_from_Edison/"+yrlong+".nc"
    # load MPAS data
    print("Loading MPAS data...")
    f=MFDataset(MPASdataFile)
    MPASdepth=np.mean(f.variables["zMid"][:,:,:],axis=0)
    MPASTemperature=np.mean(f.variables["temperature"][:,:,:],axis=0)
    MPASTemperature[np.where(MPASTemperature<-1.e3)]=np.nan
    dm=np.shape(MPASTemperature)
    nMPASCells=dm[0]
    nMPASDepth=dm[1]
    # for each transect location, generate some transects (call to transectGenerator)
    print("Calling transect generator...")
    for t in range(nTransects):
	print("")
	print("***Generating transect "+str(t)+"***")
	print("SOSE transect:")
	SOSEtransect,SOSEx,SOSEy,SOSEtransectHasData=transectGenerator(transectLonLats[t],SOSElon,SOSElat,SOSETemperature,SOSEdepth,nSOSECells,nSOSEDepth,nTransectResolution,nTransectDepths)
	print("MPAS transect:")
	MPAStransect,MPASx,MPASy,MPAStransectHasData=transectGenerator(transectLonLats[t],MPASlon,MPASlat,MPASTemperature,MPASdepth,nMPASCells,nMPASDepth,nTransectResolution,nTransectDepths)

	# calculate difference between transects.  Since coastlines differ between input datasets, *transectHasData
	# is used to identify where data exists for both datasets, and differences can be calculated
	diff=np.zeros((nTransectDepths,ntransectPoints[t]))
	diff[:,:]=np.nan
	minIndex=0
	minIndexset=False
	for n in range(ntransectPoints[t]):
            iSOSE=np.where(SOSEtransectHasData==n)
	    iMPAS=np.where(MPAStransectHasData==n)
	    if np.asarray(iSOSE).size+np.asarray(iMPAS).size==2: #if both transects have same index
	       if minIndexset==False:
		   minIndex=n
		   minIndexset=True
	       diff[:,n]=np.squeeze(MPAStransect[:,iMPAS])-np.squeeze(SOSEtransect[:,iSOSE])
	diff=diff[:,minIndex:-1]
	dm=np.shape(diff)
	ntransectPoints_trimmed=dm[1]

	# do some plotting.
	minColor=-1.5
	maxColor=1.5
	levels=np.linspace(minColor,maxColor,50)
	fig=plt.figure()
	fig.set_size_inches(11,8)
	ax1=plt.subplot2grid((10,3),(0,0),rowspan=2,colspan=3)
	cs=ax1.contourf(SOSEx,SOSEy,SOSEtransect,levels,cmap=plt.get_cmap("coolwarm"),extend="both")
	ticks_format=ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000.))           #make axis km scale, P1
	ax1.xaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P2
	ax1.yaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P3
	ax1.set_xlabel('Distance (S to N, km)')
	ax1.set_ylabel('Depth (km)')
	ax1.set_title('SOSE transect')
	cbar=fig.colorbar(cs,ax=ax1,ticks=[minColor, 0, maxColor])
	cbar.ax.set_ylabel('Temperature (C)', rotation=270)

	ax2=plt.subplot2grid((10,3),(3,0),rowspan=2,colspan=3)
	cs=ax2.contourf(MPASx,MPASy,MPAStransect,levels,cmap=plt.get_cmap("coolwarm"),extend="both")
	ticks_format=ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000.))           #make axis km scale, P1
	ax2.xaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P2
	ax2.yaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P3
	ax2.set_xlabel('Distance (S to N, km)')
	ax2.set_ylabel('Depth (km)')
	ax2.set_title('MPAS transect')
	cbar=fig.colorbar(cs,ax=ax2,ticks=[minColor, 0, maxColor])
	cbar.ax.set_ylabel('Temperature (C)', rotation=270)    

	xi=np.arange(ntransectPoints_trimmed)*nTransectResolution
	yi=np.linspace(-3000,0,nTransectDepths)
	ax3=plt.subplot2grid((10,3),(6,0),rowspan=2,colspan=3)
	cs=ax3.contourf(xi,yi,diff,levels,cmap=plt.get_cmap("coolwarm"),extend="both")
	ticks_format=ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000.))           #make axis km scale, P1
	ax3.xaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P2
	ax3.yaxis.set_major_formatter(ticks_format)                                         #make axis km scale, P3    
	cbar=fig.colorbar(cs,ax=ax3,ticks=[minColor, 0, maxColor])
	cbar.ax.set_ylabel('Temperature (C)', rotation=270)
	ax3.set_xlabel('Distance (S to N, km)')
	ax3.set_ylabel('Depth (km)')
	ax3.set_title('Difference')

	lonTransect,latTransect=zip(*transectLonLats[t]) #unpack lon,lat lists
	lonTransect=np.asarray(lonTransect)+pi #convert to #0->2pi
	latTransect=np.asarray(latTransect)
	ax4=plt.subplot2grid((10,3),(9,0),colspan=3)
	m=Basemap(llcrnrlon=0,llcrnrlat=-85,urcrnrlon=360,urcrnrlat=-50,resolution='l',ax=ax4)
	m.drawcoastlines(linewidth=0.25)
	m.fillcontinents(color='black',lake_color='aqua')
	#draw the edge of the map projection region (the projection limb)
	m.drawmapboundary(fill_color='aqua')
	#draw lat/lon grid lines every 30 degrees.
	m.drawmeridians(np.arange(0,360,30))
	m.drawparallels(np.arange(-90,90,30))
	x,y=m([lonTransect[0]/d2r, lonTransect[-1]/d2r], [latTransect[0]/d2r, latTransect[-1]/d2r])
	m.plot(x,y,'r')

	plt.savefig("/lustre/scratch2/turquoise/jer/figs/year_"+yrlong+"_transect."+'%03d'%t+".png",bbox_inches='tight')
	plt.close(fig)


