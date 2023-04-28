import numpy as np
from scipy.sparse import issparse
from scipy.ndimage import maximum_filter

## Taken from imreconstruct.cc-tst, no licence information
## Using maximum_filter instead of imdilate (which was not found)
## maximun_filter seems to behave the same way as Matlab's imdilate
def imreconstruct(marker, mask, conn):
    enter = True
    while (enter or (not np.all(marker == previous))):
        enter = False
        previous = marker
#        marker = imdilate(marker,conn)
        marker = maximum_filter(marker,footprint=conn)
        if (marker.dtype.kind == 'bool'):
            marker = (marker and mask)
        else:
            marker = np.minimum(marker,mask)
    return marker
  



def iptcheckconn(conn):
    if not isinstance(conn,np.ndarray):
        raise ValueError("Connectivity must be of type numpy.ndarray")
    else:
        dim = conn.ndim
        valid = True
        for i in range(dim):
            if (conn.shape[i]!=3):
                valid = False
        if not valid:
            raise TypeError("Connectivity must be an array with all dimensions of size 3")
        
        if (conn.dtype.kind not in np.typecodes["AllInteger"]):
            raise ValueError("Connectivity must be an array of integers")
        elif ( (conn[(1,)*dim] != 1) or ( (np.unique(conn)!=[0, 1]) and (np.unique(conn)!=[1]) )):
            raise ValueError("Connectivity must be an array with only 0 or 1 as values, and 1 at its center")
        


        
## Copyright (C) 2017 Hartmut Gimpel <hg_code@gmx.de>
def imhmin(im, h, conn=None):
    """ @deftypefn  {Function File} {} @ imhmin (@var{im}, @var{h})
        @deftypefnx {Function File} {} @ imhmin (@var{im}, @var{h}, @var{conn})
        Caculate the morphological h-minimum transform of an image @var{im}.
    
        This function removes all regional minima in the grayscale image @var{im} whose depth
        is less or equal to the given threshold level @var{h}, and it increases the depth of
        the remaining regional minima by the value of @var{h}. (A "regional minimum" is
        defined as a connected component of pixels with an equal pixel value that is less
        than the value of all its neighboring pixels. And the "depth" of a regional minimum
        can be thought of as minimum pixel value difference between the regional minimum and
        its neighboring maxima.)
    
        The input image @var{im} needs to be a real and nonsparse numeric array (of any
        dimension), and the height parameter @var{h} a non-negative scalar number.
    
        The definition of "neighborhood" for this morphological operation can be set with the
        connectivity parameter @var{conn}, which defaults to 8 for 2D images, to 26 for 3D 
        images and to @code{conn(ndims(n), "maximal")} in general. @var{conn} can be given as
        scalar value or as a boolean matrix (see @code{conndef} for details).
    
        The output is a transformed grayscale image of same type and
        shape as the input image @var{im}.
    
        @seealso{imhmax, imregionalmin, imextendedmin, imreconstruct}
        @end deftypefn
        
        Algorithm:
        * The 'classical' reference for this morphological h-minimum function is the book
           "Morphological Image Analysis" by P. Soille (Springer, 2nd edition, 2004), chapter 
           6.3.4 "Extended and h-extrema".
           It says: "This [h-maximum] is achieved by performing the reconstruction by dilation
                         of [a grayscale image] f from f-h:
                         HMAX_h(f) = R^delta_f (f - h)
                [...]    The h-minima [...] transformations are defined by analogy:
                         HMIN_h(f) = R^epsilon_f (f + h)".
        * A more easily accessible reference is for example the following
           web page by Régis Clouard:
           https://clouard.users.greyc.fr/Pantheon/experiments/morphology/index-en.html#extremum
           It says: "It is defined as the [morphological] reconstruction by erosion
                         of [a grayscale image] f increased by a height h."
          (We will call the grayscale image im instead of f.)
    """
    
    ## retrieve input parameters, set default value:
    if conn==None:
        conn = np.ones((3,)*im.ndim)
    else:
        iptcheckconn(conn)
      
    ## check input parameters:
    if ( (not isinstance(im,np.ndarray)) or (im.dtype.kind not in np.typecodes["AllFloat"]) or issparse(im) ):
        raise TypeError("imhmin: IM must be a real and nonsparse numeric array")
        
    if ((type(h)!=int) and (type(h)!=float)):
        raise TypeError("imhmin: H must be a non-negative scalar number")
    elif (h<0):
        raise ValueError("imhmin: H must be non-negative")
      
    ## do the actual calculation:
    ## (Calculate dilations of the inverse image, instead of erosions of the
    ##  original image, because this is what imreconstruct can do.
    ##  Note: "imcomplement(im)-h" is the inverse of "im+h".)
#    im = imcomplement(im)
    im2 = imreconstruct((im-h), im, conn)
    im2 = 1 - im2 # imcomplement(im2)
    
    return im2

