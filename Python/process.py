#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:50:41 2018

@author: niek
"""
import numpy as np
import gdal as gd
from osgeo import ogr
import cv2
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import math
import shapely.geometry as geometry
from shapely.geometry import Point, LineString
import centerline
from skimage.measure import compare_ssim
import imutils
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#import geopandas as gp
#import shapely
#from pyproj import Proj, transform
#import numpy as np
#from pyzbar.pyzbar import decode
#import json
#import geojson
#from sklearn.linear_model import LinearRegression
#from PIL import Image, ImageChops

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


base = cv2.imread("../Input/Base.png")
design = cv2.imread("../Input/line.png")
#
base_size = base.shape
design_size = design.shape
design_res = cv2.resize(design,(base_size[1],base_size[0]))
base_res = cv2.resize(base,(design_size[1],design_size[0]))

absdiff=cv2.absdiff(base_res,design)
mask = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)

Z=absdiff.reshape((-1,3))
Z= np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((design.shape))
res3=res2[:,:,0]-res2[:,:,2]
plt.imshow(res3==163)


grayA = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(design_res, cv2.COLOR_BGR2GRAY)
grayC = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#
#bas_arr = np.array(base)
#des_arr = np.array(design_res)

# load the image 
image = cv2.imread("../Input/line.tiff")

# load raster and define pixel coordinates
ds = gd.Open("../Input/line.tiff")
upx, xres, xskew, upy, yskew, yres = ds.GetGeoTransform()
ysiz,xsiz,bands = image.shape
xrange = np.arange(upx,upx+xsiz*xres,xres)
yrange = np.arange(upy,upy+ysiz*yres,yres)

#bbox = [float(i) for i in decode(image)[0][0].split(b'bbox=')[-1].split(b',')]
#whitex = image.shape[1]-(decode(image)[0][2].left+decode(image)[0][2].width)
#whitey = image.shape[0]-(decode(image)[0][2].top+decode(image)[0][2].height)
#dx = abs(bbox[2]-bbox[0])/(image.shape[1]-2*whitex)
#dx = abs(bbox[3]-bbox[1])/(image.shape[0]-2*whitey)
#bbox[0],bbox[2] = bbox[0]-whitex*dx , bbox[2]+whitex*dx 
#epsg = 4326
#(xs,ys) = (np.linspace(bbox[0],bbox[2],np.size(image,0)),\
#          np.linspace(bbox[1],bbox[3],np.size(image,1)))


# define the list of boundaries (this statistic can be defined by using a legend
# box to be filled by user in the future)
boundaries = [
	([0, 0, 219],[150, 160, 250])
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
    
red = output[...,2]
#red_img = Image.fromarray(red)
#red_img.show()
def extract_feature(array='2d array',value_to_extract='integer beteween 0 and 255',outname='string'):
    coords = np.argwhere(array==value_to_extract).astype(float)
    new_points = [Point(coord) for coord in coords]
    alpha = .4
    concave_hull, edge_points = alpha_shape(new_points,alpha=alpha)
    if concave_hull.geom_type == 'MultiPolygon':
        polys = {p.area:p for p in concave_hull}
        poly = polys[max(polys.keys())]
    else:
        poly = concave_hull
    
    coords = np.array([list(coord) for coord in poly.exterior.coords])
    #coords[...,1]=bbox[0]+((coords[...,1]-whitex*0.5)*(abs(bbox[2]-bbox[0])/(image.shape[1]-whitex)))
    #coords[...,0]=bbox[3]-((coords[...,0]-whitey*1.5)*(abs(bbox[3]-bbox[1])/(image.shape[0]-whitey*2)))
    new_coords = []
    for r,c in coords:
        new_coords.append([xrange[int(c)],yrange[int(r)]])
    
    vertices = [Point(point) for point in new_coords]
    
    poly = geometry.Polygon([[p.x, p.y] for p in vertices])
    line=centerline.Centerline(poly,10)
    
    ogr_poly = ogr.CreateGeometryFromWkt(poly.wkt)
    ogr_line = ogr.CreateGeometryFromWkt(line.wkt)
    
    src_srs = ogr.osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    
    dest_srs = ogr.osr.SpatialReference()
    dest_srs.ImportFromEPSG(4326)
    
    coordTransform = ogr.osr.CoordinateTransformation(src_srs, dest_srs)
    ogr_poly.Transform(coordTransform)
    ogr_line.Transform(coordTransform)
    
    with open('../Output/line_'+outname+'.geojson','w') as of:
        of.write(ogr_line.ExportToJson())
    of.close()
    
    with open('../Output/polygon_'+outname+'.geojson','w') as of:
        of.write(ogr_poly.ExportToJson())
    of.close()
    
extract_feature(red>0,1,'red')
res3values = np.unique(res3)
print(res3values)
extract_feature(res3,res3values[3],'k_means_'+str(res3values[3]))
    


## To do --> different approach to extracting feature with alpha shape and then 
## defining if polygon or line is most suitable...
