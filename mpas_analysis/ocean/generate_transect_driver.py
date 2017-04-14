import numpy as np
import scipy.io
import math
from netCDF4 import MFDataset
from pyproj import Geod
from transect_generator import transectGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker

d2r=math.radians(1.)
pi=math.pi

print("Loading SOSE data...")
vnames=['THETA_2005','THETA_2006','THETA_2007','THETA_2008','THETA_2009','THETA_2010']
vnames=['THETA_2005']
nSOSECells=2160*320
nSOSEDepth=42
SOSETemperature=np.zeros((nSOSECells,nSOSEDepth,len(vnames)))
for n,f in enumerate(vnames):
   arr=scipy.io.loadmat('/lustre/scratch2/turquoise/jer/THETA_AnnualAvg.mat')[f]
   SOSETemperature[:,:,n]=arr.reshape(-1,arr.shape[-1])
SOSETemperature=np.mean(SOSETemperature,axis=2)
SOSETemperature[np.where(SOSETemperature==0.0)]=np.nan
SOSElon=np.ravel(scipy.io.loadmat("/lustre/scratch2/turquoise/jer/SOSEgrid.mat")['XC'])*d2r #degrees, 0->2pi
SOSElat=np.ravel(scipy.io.loadmat("/lustre/scratch2/turquoise/jer/SOSEgrid.mat")['YC'])*d2r #degrees, -pi/2->pi/2
SOSEdepth=scipy.io.loadmat("/lustre/scratch2/turquoise/jer/SOSEgrid.mat")['RC'] #m, 0->-5575
SOSEdepth=np.tile(np.transpose(SOSEdepth),(nSOSECells,1))

print("Loading MPAS data...")
infile="/users/jer/playground/test_G_B_static_compsets/B_build_double_counting/run/mpaso.hist.0009-*.nc"
#infile="/users/jer/playground/test_G_B_static_compsets/B_build_double_counting/run/mpaso.hist.0009-12-01_00000.nc"
f=MFDataset(infile)
MPASdepth=np.mean(f.variables["zMid"][:,:,:],axis=0)
MPASTemperature=np.mean(f.variables["temperature"][:,:,:],axis=0)
MPASTemperature[np.where(MPASTemperature<-1.e3)]=np.nan
dm=np.shape(MPASTemperature)
nMPASCells=dm[0]
nMPASDepth=dm[1]
MPASlat=f.variables["latCell"][:] #-pi/2->pi/2
MPASlon=f.variables["lonCell"][:] #0->2pi

print("Making transects...")
g=Geod(ellps='sphere')
nTransectResolution=20000. #m
nTransectDepths=300
transectDistance=list()
transectLonLats=list()
ntransectPoints=list()
endPoints=np.zeros(4)
for lon in np.arange(-175,175,2): #start lat, start lon, end lat, end lon.  ***Must be monotonically increasing or constant lat & lon values
#for lon in [0]: #start lat, start lon, end lat, end lon.  ***Must be monotonically increasing or constant lat & lon values
    endPoints[0]=-80.
    endPoints[1]=lon
    endPoints[2]=-60.
    endPoints[3]=lon
    endPoints[:]=endPoints[:]*d2r
    #Construct transect points in a great circle from start to end lat/lon points
    d=g.inv(endPoints[1],endPoints[0],
            endPoints[3],endPoints[2],radians=True)
    nt=np.floor(d[2]/nTransectResolution).astype(int)
    transectPoints=g.npts(endPoints[1],endPoints[0],
                          endPoints[3],endPoints[2],nt,radians=True)
    ntransectPoints.append(nt)
    transectDistance.append(d[2])
    transectLonLats.append(transectPoints)
nTransects=len(transectDistance)

print("Calling transect generator...")
for t in range(nTransects):
    SOSEtransect,SOSEx,SOSEy,SOSEtransectHasData=transectGenerator(transectLonLats[t],SOSElon,SOSElat,SOSETemperature,SOSEdepth,nSOSECells,nSOSEDepth,nTransectResolution,nTransectDepths)
    MPAStransect,MPASx,MPASy,MPAStransectHasData=transectGenerator(transectLonLats[t],MPASlon,MPASlat,MPASTemperature,MPASdepth,nMPASCells,nMPASDepth,nTransectResolution,nTransectDepths)

    diff=np.zeros((nTransectDepths,ntransectPoints[t],))
    diff[:,:]=np.nan
    for n in range(ntransectPoints[t]):
        iSOSE=np.where(SOSEtransectHasData==n)
	iMPAS=np.where(MPAStransectHasData==n)
	if np.asarray(iSOSE).size+np.asarray(iMPAS).size==2: #if both transects have same index
	   diff[:,n]=np.squeeze(SOSEtransect[:,iSOSE])-np.squeeze(MPAStransect[:,iMPAS])
	
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

    ###DUPLICATE CODE TO REMOVE
    lonTransect,latTransect=zip(*transectLonLats[t]) #unpack lon,lat lists
    lonTransect=np.asarray(lonTransect)+pi #convert to #0->2pi
    latTransect=np.asarray(latTransect)
    ###DUPLICATE CODE TO REMOVE
    
    xi=np.arange(ntransectPoints[t])*nTransectResolution
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

    ###DUPLICATE CODE TO REMOVE
    lonTransect,latTransect=zip(*transectLonLats[t]) #unpack lon,lat lists
    lonTransect=np.asarray(lonTransect)+pi #convert to #0->2pi
    latTransect=np.asarray(latTransect)
    ###DUPLICATE CODE TO REMOVE

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
    
    plt.savefig("/lustre/scratch2/turquoise/jer/figs/transect."+'%03d'%t+".png",bbox_inches='tight')
    plt.close(fig)


