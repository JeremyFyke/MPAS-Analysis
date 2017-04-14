import numpy as np
from pyproj import Geod, Proj
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import griddata
import math
d2r=math.radians(1.)
pi=math.pi
def transectGenerator(transectLonLats,lon,lat,v,z,nCells,nDepth,nTransectResolution,nTransectDepths):
    lonTransect,latTransect=zip(*transectLonLats) #unpack lon,lat lists
    lonTransect=np.asarray(lonTransect)+pi #convert to #0->2pi
    latTransect=np.asarray(latTransect)
    lats=latTransect[0]
    lons=lonTransect[0]
    late=latTransect[-1]
    lone=lonTransect[-1]
    nTransectPoints=np.size(lonTransect)
    transectHasData=np.arange(nTransectPoints) #Vector with indices to transect points
    print("Start/end transect lon:"+str(lons)+"/"+str(lone))
    print("Start/end transect lat:"+str(lats)+"/"+str(late))
    print("Min/max lon:"+str(np.min(lon))+"/"+str(np.max(lon)))
    print("Min/max lat:"+str(np.min(lat))+"/"+str(np.max(lat)))
    InterpBuffer=1.*d2r #1 degree, in radians
    i=np.where((lon>=lons-InterpBuffer) &
               (lon<=lone+InterpBuffer) &
               (lat>=lats-InterpBuffer) &
               (lat<=late+InterpBuffer))
    print("Total # of interpolant points: "+str(np.size(i)))
    #Get minimu/maximum range of ocean values
    latmin=np.min(lat[i])
    latmax=np.max(lat[i])
    latmid=np.mean([latmin,latmax])
    lonmin=np.min(lon[i])
    lonmax=np.max(lon[i])
    lonmid=np.mean([lonmin,lonmax])
    #Make regional projection here
    pgrid=Proj(proj='stere',lat_0=latmid/d2r,lon_0=lonmid/d2r, ellps="sphere")
    
    print("Min/max near-transect lon:"+str(lonmin)+"/"+str(lonmax))
    print("Min/max near-transect lat:"+str(latmin)+"/"+str(latmax))
    print("Start/end transect lon:"+str(lons)+"/"+str(lone))
    print("Start/end transect lat:"+str(lats)+"/"+str(late))    
    #Find transect points that fall within min/max lat/lons
    print("Total # of interpolant points prior to coastal trimming: "+str(np.size(latTransect)))
    ii=np.where((latTransect>latmin) & 
                (latTransect<latmax) &
		(lonTransect>lonmin) &
		(lonTransect<lonmax))
    lonTransect=lonTransect[ii]
    latTransect=latTransect[ii]
    transectHasData=transectHasData[ii] #Indices with no-data points removed
    nTransectPoints=np.size(lonTransect) #Revise number of transect points
    
    print("Total # of interpolant points after coastal trimming: "+str(nTransectPoints))
    x,y=pgrid(lon[i]/d2r,lat[i]/d2r)
    points=np.column_stack((x,y))
    xTransect,yTransect=pgrid(lonTransect/d2r,latTransect/d2r)
    transect=np.zeros((nDepth,nTransectPoints))
    depth=np.zeros((nDepth,nTransectPoints))
    for d in range(nDepth): #iterate over depths
        VariableInterpolator=LinearNDInterpolator(points,np.squeeze(v[i,d]))
	DepthInterpolator=LinearNDInterpolator(points,np.squeeze(z[i,d]))
	transect[d,:]=VariableInterpolator(xTransect,yTransect)
	depth[d,:]=DepthInterpolator(xTransect,yTransect)
    z=np.any(np.isnan(depth),axis=0)
    depth=depth[:,-z]
    transect=transect[:,-z]
    transectHasData=transectHasData[-z] #Indices with no data removed
    dm=np.shape(transect)
    nTransectPoints=dm[1]
    print("Total # of interpolant points after NaN trimming: "+str(nTransectPoints))

    xi=np.arange(nTransectPoints)*nTransectResolution #along-transect points
    yi=np.linspace(-3000,0,nTransectDepths) #depth
    grid_x,grid_y=np.meshgrid(xi,yi)
    x=np.tile(xi,(nDepth,1))
    transect_gridded=griddata((np.ravel(x),np.ravel(depth)),np.ravel(transect),(grid_x,grid_y),method='linear')

    return transect_gridded,xi,yi,transectHasData
