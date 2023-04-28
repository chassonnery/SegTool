"""
// -------------------------------------------------- //
//                                                    //
//           **ClusterizeSphereObjects :**            //
//        Automation Script for the clustering        //
//        of spherical objects using watershed        //
//                                                    //
// -------------------------------------------------- //
// **Original algorithm :**                           //
//                                                    //
// **Script developers :**                            //
//   Pauline CHASSONNERY                              //
// -------------------------------------------------- //


## In case you use the results of this script in your article, please don't forget to cite us:
****************

## Purpose: *****************


## Copyrights (C) ***********

## License:
ClusterizeSphereObjects is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 
https://www.gnu.org/licenses/gpl-3.0.en.html 

## Commercial use:
The GPLv3 license cited above does not permit any commercial (profit-making or proprietary) use or re-licensing or re-distributions. Persons 
interested in for-profit use should contact the author. Note that the commercial use of this script is also protected by patent number: *******

"""
# Version: 2022-08-01

import numpy as np
import pandas as pd
import math as math
import scipy.ndimage as scim
from skimage.morphology import local_maxima, remove_small_objects
from skimage.segmentation import watershed
from functions_fromMatlab_viaOctave import imhmin
from warnings import warn


# ClusterizeSpheresUsingWatershed
def ClusterizeSphereObjects(InputSpheres, InputRods=None, PeriodicBoundaryCondition=False, xmax=None, ymax=None, zmax=None, dil_coeff=1,\
                            resolution=10, smooth_coeff=3, MinNumberOfObjectInCluster=5, header="Cluster"): 
    """ *********************
    
    Summary
    -------
    ***************** < Copy purpose here >
    
    
    Parameters
    ----------
    InputSpheres : pandas.DataFrame or str
        Either a DataFrame containing data relative to/ describing a set of spherical objects or a string path leading to a csv file from which to
        retrieve the data.
        The DataFrame or file must have at least 2 rows and 4 columns and is assumed to be formatted as follow : 
            - one row per object,
            - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the sphere's center,
            - one column with label/header "R" which contains the sphere's radius. 
        Extra/additional columns are accepted but will not be used.
    
    InputRods : pandas.DataFrame or str or None, optional
        Either a DataFrame containing data relative to/ describing a set of spherocylindrical (rod-like) objects or a string path leading to a csv
        file from which to retrieve the data or None if there is no rod-like object in the system.
        The DataFrame or file must have at least 1 rows and 4 columns and is assumed to be formatted as follow :
            - one row per object,
            - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the spherocylinder's center,
            - three columns with label/header "wX", "wY" and "wZ" which contain its orientation vector,
            - one column with label/header "L" which contains its length (i.e. length of the central cylindrical part),
            - one column with label/header "R" which contains its radius.
        Extra/additional columns are accepted but will not be used.
        Default is None.
    
    PeriodicBoundaryCondition : bool, optional 
        Specify if the domain boundary conditions are periodic or not. Default is False.
    
    xmax : float, optional
        Half-length of the computation domain in the x direction. User-provided value is only needed if ``PeriodicBoundaryCondition`` is True, 
        otherwise default value is alright. Default is computed as the smallest value allowing to enclose all the spheres. See corresponding section 
        in the manual for more details.
        If ``PeriodicBoundaryCondition`` is True but no value is provided for ``xmax``, computation will run with default value but a warning will be 
        issued to the user.
    
    ymax : float, optional
        Half-length of the computation domain in the y direction (identical to xmax).
    
    zmax : float, optional
        Half-length of the computation domain in the z direction (identical to xmax).
    
    dil_coeff : float, optional
        Dilatation coefficient to be applied to the spheres' radius to make clustering easier. See corresponding section in the manual for more
        details. Default value is 1 (i.e. no dilatation).
    
    resolution : int, optional
        Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere. Default value is 10.
    
    smooth_coeff : float, optional
        Smoothing coefficient applied to the image (via a h-minima transformation) before watershed. See corresponding section in the manual for more 
        details. Default value is 3.
    
    MinNumberOfObjectInCluster : int, optional
        Minimal number of objects a cluster must contain to be considered as valid. Default value is 5.
    
    header : str, optional
        Header to be given to the column containing the result of the clusterization process. Default value is "Cluster".
    
    
    References
    ----------
        The manual *****************
    
    
    Examples
    --------
    example 1
        import pandas as pd
        from WatershedClustering import ClusterizeSphereObjects
        
        Results = ClusterizeSphereObjects("demo_data_spheres.csv", InputRods="demo_data_rods.csv", dil_coeff=1.2)
        
        disp(Results)
        
        Results.to_csv("my_file.csv")
    
    example 2
        import pandas as pd
        from WatershedClustering import ClusterizeSphereObjects
        import matplotlib.pyplot as plt
        
        Results = ClusterizeSphereObjects("demo_data_spheres.csv", header="cluster (test 1)")
        Results = ClusterizeSphereObjects(Results, dil_coeff=1.2, header="cluster (test 2)")
        Results = ClusterizeSphereObjects(Results, InputRods="demo_data_rods.csv", dil_coeff=1.2, header="cluster (test 3)")
        Results = ClusterizeSphereObjects(Results, InputRods="demo_data_rods.csv", PeriodicBoundaryCondition=True, xmax=20, ymax=10, zmax=10, \
                                          dil_coeff=1.2, header="cluster (test 4)")
        
        plt.figure(1)
        plt.subplot(1,4,1)
        plt.plot3()
        plt.title("Clusterization with default parameters")

        plt.subplot(1,4,2)
        plt.plot3()
        plt.title("Clusterization with dilation by coefficient 1.2")

        plt.subplot(1,4,3)
        plt.plot3()
        plt.title("Clusterization with dilation by coefficient 1.2 \n and using rod objects as separators")

        plt.subplot(1,4,4)
        plt.plot3()
        plt.title("Clusterization with periodic boundary conditions enabled")
        
        
        # ## Save watershed output as tiff file
        # norm = matplotlib.colors.Normalize(vmin=np.min(labels3), vmax=np.max(labels3), clip=True)
        # mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.nipy_spectral)
        # 
        # imstack = []
        # for z in range(labels3.shape[2]):
        #     RGBA_data = mapper.to_rgba(labels3[:,:,z])*255
        #     RGBA_data = RGBA_data.astype(np.uint8)
        #     imstack.append(Image.fromarray( RGBA_data[:,:,:3],mode='RGB' )) # drop last coordinate, that is transparency
        # 
        # imstack[0].save('Python_watershed_cv.tiff',save_all=True,append_images=imstack[1:])
    """
    

    # Retrieve data on the set of spheres and check its validity
    SpheresSet = RetrieveSpheresData(InputSpheres)
    
    # Retrieve data on the set of rods and check its validity
    RodsSet = RetrieveRodsData(InputRods)
    
    # Instanciate of the class ParametersForClustering (check validity of user-defined parameters and compute internal parameters)
    param = ParametersForClustering(SpheresSet, RodsSet, PeriodicBoundaryCondition, xmax, ymax, zmax, dil_coeff, resolution, smooth_coeff,\
                                    MinNumberOfObjectInCluster)
    
    # Check validity of parameter 'header'
    if not isinstance(header,str):
        raise TypeError("header is the name for the column containing the result of the clusterization process, it must be a string.")
    
    # Create a 3D binary image with 1's for spheres and 0's for background
    bw = Create3Dbinaryimage(param,SpheresSet,RodsSet)
    
    # Apply watershed algorithm to this image (with standard preprocess)
    ClusterMap = MapRegionsUsingWatershed(bw,param.pixelsize,param.smooth_coeff)
    
    # Identify the cluster index attributed to each element of 'SpheresSet' according to the cluster-map obtained by watershed
    ClusterIndex = ClusterizeFromMap(param,SpheresSet,ClusterMap)
    
    # Add to 'SpheresSet' a column containing the cluster index of each element
    SpheresSet[header] = pd.Series(ClusterIndex)
    
    # # Save the updated DataFrame 'SpheresSet' (original data plus result of the clusterization process) to a csv file
    # SpheresSet.to_csv(filename)
    
    return SpheresSet



def ClusterizeBinaryImage(InputBW, pixelsize=[1,1,1], smooth_coeff=3, MinNumberOfPixelInCluster=1): # MinSizeOfObject
    """
    """

    # Retrieve binary map from a tiff file or a ndarray and check its validity
    BinaryArray = RetrieveBinaryData(InputBW)
    
    # Remove isolated spots/objects too small to constitute a cluster
    BinaryArray = remove_small_objects(BinaryArray, min_size=MinNumberOfPixelInCluster, connectivity=3)
    
    # Apply watershed algorithm to this image (with standard preprocess)
    ClusterMap = MapRegionsUsingWatershed(BinaryArray,pixelsize,smooth_coeff)
    
    return ClusterMap



def RetrieveBinaryData(InputBW):
    """Check that InputBW is well formatted. If formatted as a numpy.ndarray, copy it. If formatted as a filename, read data from this file. 
    """
    # If the user provided the data as a ndarray, make a copy of this ndarray.
    if isinstance(InputBW,np.ndarray):
        BinaryArray = np.copy(InputBW)
    # If the user provided a filename, read the data from this file.
    elif isinstance(InputBW,str):
        BinaryImage = Image.open(InputBW)
        BinaryArray = []
        for frame in range(BinaryImage.n_frames):
            BinaryImage.seek(i)
            BinaryArray.append(np.array(BinaryImage))
        BinaryArray = np.array(BinaryArray)
    # Otherwise protest.
    else:
        raise TypeError("InputBW must be either a *numpy.ndarray* containing the data or a *string* leading to a tiff file from which to \
                         retrieve the data.")  
    
    # Check that the array is binary, else binarize
    ...
    
    return BinaryArray
    

    
def RetrieveSpheresData(InputSpheres):
    """Check that InputSpheres is well formatted. If formatted as a Dataframe, copy it. If formatted as a filename, read data from this file. 
    """
    # If the user provided the data as a DataFrame, make a copy of this DataFrame.
    if isinstance(InputSpheres,pd.DataFrame):
        SpheresSet = InputSpheres.copy(deep=True)
    # If the user provided a filename, read the data from this file.
    elif isinstance(InputSpheres,str):
        SpheresSet = pd.read_csv(InputSpheres)
    # Otherwise protest.
    else:
        raise TypeError("InputSpheres must be either a *pandas.DataFrame* containing the data or a *string* leading to a file from which to retrieve \
                         the data.")  
    
    # Check if the DataFrame contains the necessary columns
    if "X" not in SpheresSet.columns:
        raise ValueError("Missing column 'X' in InputSpheres !")
    elif ((SpheresSet["X"].dtypes.kind not in np.typecodes["AllFloat"]) and (SpheresSet["X"].dtypes.kind=='i')):
        raise TypeError("Column 'X' of InputSpheres must contain scalar numbers" )

    if "Y" not in SpheresSet.columns:
        raise ValueError("Missing column 'Y' in InputSpheres !")
    elif ((SpheresSet["Y"].dtypes.kind not in np.typecodes["AllFloat"]) and (SpheresSet["Y"].dtypes.kind=='i')):
        raise TypeError("Column 'Y' of InputSpheres must contain scalar numbers" )

    if "Z" not in SpheresSet.columns:
        raise ValueError("Missing column 'Z' in InputSpheres !")
    elif ((SpheresSet["Z"].dtypes.kind not in np.typecodes["AllFloat"]) and (SpheresSet["Z"].dtypes.kind=='i')):
        raise TypeError("Column 'Z' of InputSpheres must contain scalar numbers" )

    if "R" not in SpheresSet.columns:
        raise ValueError("Missing column 'R' in InputSpheres !")
    elif ((SpheresSet["R"].dtypes.kind not in np.typecodes["AllFloat"]) and (SpheresSet["R"].dtypes.kind=='i')):
        raise TypeError("Column 'R' of InputSpheres must contain scalar numbers" )
    
    if (SpheresSet.shape[0] < 2):
        raise ValueError("InputSpheres must have at least 2 rows (that is it must describe at least 2 objects) !")
    
    return SpheresSet



def RetrieveRodsData(InputRods):
    """Check that InputRods is well formatted. If formatted as a Dataframe, copy it. If formatted as a filename, read data from this file.
    """
    # If the user did not provide data, then their is no rod-like objects in the system
    if InputRods==None:
        RodsSet = None
    # If the user provided data as a DataFrame, make a copy of this DataFrame.
    elif isinstance(InputRods,pd.DataFrame):
        RodsSet = InputRods.copy(deep=True)
    # If the user provided a filename, read the data from this file.
    elif isinstance(InputRods,str):
        RodsSet = pd.read_csv(InputRods)
    # Otherwise protest.
    else:
        raise TypeError("InputRods must be either *None* (meaning there is no rod-like object in the system), a *pandas.DataFrame* containing the \
                          data or a *string* leading to a file from which to retrieve the data.")
    
    # Check if the DataFrame contains the necessary columns
    if RodsSet!=None :
        if "X" not in RodsSet.columns:
            raise ValueError("Missing column 'X' in InputRods !")
        elif ((RodsSet["X"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["X"].dtypes.kind=='i')):
            raise TypeError("Column 'X' of InputRods must contain scalar numbers" )

        if "Y" not in RodsSet.columns:
            raise ValueError("Missing column 'Y' in InputRods !")
        elif ((RodsSet["Y"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["Y"].dtypes.kind=='i')):
            raise TypeError("Column 'Y' of InputRods must contain scalar numbers" )

        if "Z" not in RodsSet.columns:
            raise ValueError("Missing column 'Z' in InputRods !")
        elif ((RodsSet["Z"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["Z"].dtypes.kind=='i')):
            raise TypeError("Column 'Z' of InputRods must contain scalar numbers" )

        if "wX" not in RodsSet.columns:
            raise ValueError("Missing column 'wX' in InputRods !")
        elif ((RodsSet["wX"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["wX"].dtypes.kind=='i')):
            raise TypeError("Column 'wX' of InputRods must contain scalar numbers" )

        if "wY" not in RodsSet.columns:
            raise ValueError("Missing column 'wY' in InputRods !")
        elif ((RodsSet["wY"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["wY"].dtypes.kind=='i')):
            raise TypeError("Column 'wY' of InputRods must contain scalar numbers" )

        if "wZ" not in RodsSet.columns:
            raise ValueError("Missing column 'wZ' in InputRods !")
        elif ((RodsSet["wZ"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["wZ"].dtypes.kind=='i')):
            raise TypeError("Column 'wZ' of InputRods must contain scalar numbers" )
        
        if "L" not in RodsSet.columns:
            raise ValueError("Missing column 'L' in InputRods !")
        elif ((RodsSet["L"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["L"].dtypes.kind=='i')):
            raise TypeError("Column 'L' of InputRods must contain scalar numbers" )
            
        if "R" not in RodsSet.columns:
            raise ValueError("Missing column 'R' in InputRods !")
        elif ((RodsSet["R"].dtypes.kind not in np.typecodes["AllFloat"]) and (RodsSet["R"].dtypes.kind=='i')):
            raise TypeError("Column 'R' of InputRods must contain scalar numbers" )
    
    return RodsSet



class ParametersForClustering:
    """This class implement and check the validity/compatibility of all the parameters needed for the Watershed Clustering algorithm. 
    
    
    Parameters
    ----------
    SpheresSet : pandas.DataFrame
        DataFrame containing data relative to/ describing a set of spherical objects. It must have at least 2 rows and 4 columns and is assumed to be 
        formatted as follow : 
            - one row per object,
            - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the sphere's center,
            - one column with label/header "R" which contains the sphere's radius. 
        Extra/additional columns are accepted but will not be used.
    
    RodsSet : pandas.DataFrame or None, optional
        Either a DataFrame containing data relative to/ describing a set of spherocylindrical (rod-like) objects or None if there is no rod-like
        object in the system. The DataFrame must have at least 1 rows and 4 columns and is assumed to be formatted as follow :
            - one row per object,
            - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the spherocylinder's center,
            - three columns with label/header "wX", "wY" and "wZ" which contain its orientation vector,
            - one column with label/header "L" which contains its length (i.e. length of the central cylindrical part),
            - one column with label/header "R" which contains its radius.
        Extra/additional columns are accepted but will not be used.
        Default is None.
    
    PeriodicBoundaryCondition : bool, optional 
        Specify if the domain boundary conditions periodic or not. Default is False.
    
    xmax : float, optional
        Half-length of the computation domain in the x direction. User-provided value is only needed if ``PeriodicBoundaryCondition`` is True,
        otherwise default value is alright. Default is computed as the smallest value allowing to enclose all the spheres. See corresponding section 
        in the manual for more details.
        If ``PeriodicBoundaryCondition`` is True but no value is provided for ``xmax``, computation will run with default value but a warning will be 
        issued to the user.
    
    ymax : float, optional
        Half-length of the computation domain in the y direction (identical to xmax).
    
    zmax : float, optional
        Half-length of the computation domain in the z direction (identical to xmax).
    
    dil_coeff : float, optional
        Dilatation coefficient to be applied to the spheres' radius to make clustering easier. See corresponding section in the manual for more 
        details. Default value is 1 (i.e. no dilatation).
    
    resolution : int, optional
        Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere. Default value is 10.
    
    smooth_coeff : float, optional
        Smoothing coefficient applied to the image (via a h-minima transformation) before watershed. See corresponding section in the manual for more 
        details. Default value is 3.
    
    MinNumberOfObjectInCluster : int
        Minimal number of objects a cluster must contain to be considered as valid. Default value is 5.
    
    
    Attributes
    ----------
    Nspheres : int
        Number of spherical objects in the system.
    Nrods : int
        Number of rod-like objects in the system.
    PeriodicBoundaryCondition : bool
        Specify if the domain boundary conditions periodic or not.
    xmax, ymax, zmax : float
        Half-length of the computation domain in the x, y and z direction respectively.
    dil_coeff : float
        Dilatation coefficient to be applied to the spheres' radius to make clustering easier.
    resolution : int
        Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere.    
    pixelsize : float
        Size of a cubic pixel, equal to the diameter of the smallest sphere divided by ``resolution``.
    xgrid : numpy.ndarray (ndim = 3)
        3D array containing a the x-grid part of a meshgrid.
        If ``PeriodicBoundaryCondition`` is False, the grid span over domain [-``xmax``,``xmax``] with a uniform step size equal to ``pixelsize``
        If ``PeriodicBoundaryCondition`` is True, the grid span over domain [-2``xmax``,2``xmax``] with a uniform step size equal to ``pixelsize``
        In case the value of ``pixelsize`` does not allow for a whole number of points in the domain described above, this domain will be slightly 
        extended on the right-hand side. 
    ygrid : numpy.ndarray (ndim = 3)
        3D array containing a the y-grid part of a meshgrid (same as xgrid).
    zgrid : numpy.ndarray (ndim = 3)
        3D array containing a the z-grid part of a meshgrid (same as xgrid).
    Nx : int
        Number of pixels of the image created for clustering in the x direction.
        If ``PeriodicBoundaryCondition`` is False then (``Nx`` - 1) x ``pixelsize`` ≥ ``xmax``.
        If ``PeriodicBoundaryCondition`` is True then (``Nx`` - 1) x ``pixelsize`` ≥ 2 x ``xmax``.
    Ny : int
        Number of pixels of the image created for clustering in the y direction (same as Nx).
    Nz : int
        Number of pixels of the image created for clustering in the z direction (same as Nx).
    smooth_coeff : float
        Smoothing coefficient applied to the image (via a h-minima transformation) before watershed.
    MinNumberOfObjectInCluster : int
        Minimal number of objects a cluster must contain to be considered as valid.   
    """
    
    def __init__(self, SpheresSet, RodsSet=None, PeriodicBoundaryCondition=False, xmax=None, ymax=None, zmax=None, dil_coeff=1, resolution=10, \
                 smooth_coeff=3, MinNumberOfObjectInCluster=5):
        """Constructor.
        """
        # Retrieve number of spherical objects
        self.Nspheres = len(SpheresSet["X"])
        
        # Retrieve number of rod-like objects
        if RodsSet == None:
            self.Nrods = 0
        else:
            self.Nrods = len(RodsSet["X"])
        
        # Copy value of parameter 'PeriodicBoundaryCondition' into the equivalent attribute
        if isinstance(PeriodicBoundaryCondition,bool):
            self.PeriodicBoundaryCondition = PeriodicBoundaryCondition
        else:
            raise TypeError("PeriodicBoundaryCondition must be of type logical (i.e. either True or False).")
        
        
        # If the user provided no value for xmax and/or ymax and/or zmax, compute them as the half-length of the smallest cuboid box centered on the
        # origin and enclosing all the spheres of the set.
        # If the user asked for periodic boundary condition but did not provide value for xmax and/or ymax and/or zmax, keep running but issue a 
        # warning.
        warning_message = "Warning : for periodic boundary conditions, auto-estimation of the domain size is not a reliable option and may lead to \
                           invalid results. Please provide value for the half-length of the computation domain in the x, y and z directions."
        warning_trigger = False
        
        if xmax == None:
            self.xmax = np.max(np.abs(SpheresSet["X"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaryCondition == True:
                warning_trigger = True
        elif (isinstance(xmax,(int,float)) and (xmax > 0)):
            self.xmax = xmax
        else:
            raise TypeError("xmax must be a positive scalar number")
        
        if ymax == None:
            self.ymax = np.max(np.abs(SpheresSet["Y"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaryCondition == True:
                warning_trigger = True
        elif (isinstance(ymax,(int,float)) and (ymax > 0)):
            self.ymax = ymax
        else:
            raise TypeError("ymax must be a positive scalar number")
        
        if zmax == None:
            self.zmax = np.max(np.abs(SpheresSet["Z"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaryCondition == True:
                warning_trigger = True
        elif (isinstance(zmax,(int,float)) and (zmax > 0)):
            self.zmax = zmax
        else:
            raise TypeError("zmax must be a positive scalar number")
            
        if warning_trigger:
            warn(warning_message) 
        
        
        # Copy value of parameter 'dil_coeff' into the equivalent attribute
        if (isinstance(dil_coeff,(int,float)) and (dil_coeff > 0)):
            self.dil_coeff = dil_coeff
        else:
            raise TypeError("dif_coeff must be a positive scalar number")
        
        # Copy value of parameter 'resolution' into the equivalent attribute
        if (isinstance(resolution,int) and (resolution > 0)):
            self.resolution = resolution
        else:
            raise TypeError("resolution must be a positive integer")
        
        # Copy value of parameter 'smooth_coeff' into the equivalent attribute
        if (isinstance(smooth_coeff,(int,float)) and (smooth_coeff >= 0)):
            self.smooth_coeff = smooth_coeff
        else:
            raise TypeError("smooth_coeff must be a non-negative scalar number")
        
        # Copy value of parameter 'MinNumberOfObjectInCluster' into the equivalent attribute
        if (isinstance(MinNumberOfObjectInCluster,int) and (MinNumberOfObjectInCluster >= 0)):
            self.MinNumberOfObjectInCluster = MinNumberOfObjectInCluster
        else:
            raise TypeError("MinNumberOfObjectInCluster must be a non-negative integer")
        
        
        # Compute value of 'pixelsize' as the diameter of the smallest sphere divided by 'resolution'.
        self.pixelsize = 2.0*min(SpheresSet["R"])/self.resolution
        
        
        # Create 3D grid spaning over domain [-2'xmax',2'xmax']x[-2'ymax',2'ymax']x[-2'zmax',2'zmax'] with a uniform step size equal to 'pixelsize'
        if self.PeriodicBoundaryCondition:
            [self.xgrid,self.ygrid,self.zgrid] = np.meshgrid(np.arange(-2*self.xmax,2*self.xmax+self.pixelsize,self.pixelsize), \
                                                             np.arange(-2*self.ymax,2*self.ymax+self.pixelsize,self.pixelsize), \
                                                             np.arange(-2*self.zmax,2*self.zmax+self.pixelsize,self.pixelsize),indexing='ij')
        # Create 3D grid spaning over domain [-'xmax','xmax']x[-'ymax','ymax']x[-'zmax','zmax'] with a uniform step size equal to 'pixelsize'
        else:
            [self.xgrid,self.ygrid,self.zgrid] = np.meshgrid(np.arange(-self.xmax,self.xmax+self.pixelsize,self.pixelsize), \
                                                             np.arange(-self.ymax,self.ymax+self.pixelsize,self.pixelsize), \
                                                             np.arange(-self.zmax,self.zmax+self.pixelsize,self.pixelsize),indexing='ij')
        
        # Compute number of pixels in the grid
        [self.Nx,self.Ny,self.Nz] = self.xgrid.shape



def __FindVirtualDuplicate(param,xin,yin,zin,per):
    """Return position of the center of the ``per``-th duplicate of an object located at (``xin``,``yin``,``zin``).
    per = 0 : real position
    per = 1 : transposition through the closest face in the x direction
    per = 2 : transposition through the closest face in the y direction
    per = 3 : transposition through the closest face in the z direction
    per = 4 : transposition through the closest edge in the xy direction
    per = 5 : transposition through the closest edge in the xz direction
    per = 6 : transposition through the closest edge in the yz direction
    per = 7 : transposition through the closest vertex
    """
    # Tranlation through the closest face in the x direction
    xper = xin - 2*math.copysign(param.xmax,xin)
    # Tranlated position of the object through the closest y-edge
    yper = yin - 2*math.copysign(param.ymax,yin)
    # Tranlated position of the object through the closest z-edge
    zper = zin - 2*math.copysign(param.zmax,zin)
    
    if per==0:
        xout = xin  
        yout = yin
        zout = zin
    elif per==1:
        xout = xper
        yout = yin
        zout = zin
    elif per==2:
        xout = xin
        yout = yper
        zout = zin
    elif per==3:
        xout = xin
        yout = yin
        zout = zper
    elif per==4:
        xout = xper
        yout = yper
        zout = zin
    elif per==5:
        xout = xper
        yout = yin
        zout = zper
    elif per==6:
        xout = xin
        yout = yper
        zout = zper
    elif per==7:
        xout = xper
        yout = yper
        zout = zper
    
    return xout,yout,zout



def __SphereMask(param,x,y,z,r):
    """Return a small binary image of a sphere of center (``x``,``y``,``z``) and radius ``r``, as well as the coordinates
    [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max] of this image in the whole picture.
    """
    # Compute the smallest box [Nx_min,Nx_max[ x [Ny_min,Ny_max[ x [Nz_min,Nz_max[ entirely containing this sphere. To help with the segmentation to
    # come, the radius of the sphere is dilated by dil_coeff. Note the left-hand inclusion and right-hand exclusion to accommodate Python indexing.
    # Grid's index are between 0 and Nx-1 (both included), so the range is cut-off to [0,Nx[.
    Nx_min = max(math.floor( (x - r + (int(param.PeriodicBoundaryCondition) + 1)*param.xmax)/param.pixelsize )    , 0)
    Nx_max = min(math.floor( (x + r + (int(param.PeriodicBoundaryCondition) + 1)*param.xmax)/param.pixelsize ) + 2, param.Nx)
    
    Ny_min = max(math.floor( (y - r + (int(param.PeriodicBoundaryCondition) + 1)*param.ymax)/param.pixelsize )    , 0)
    Ny_max = min(math.floor( (y + r + (int(param.PeriodicBoundaryCondition) + 1)*param.ymax)/param.pixelsize ) + 2, param.Ny)
    
    Nz_min = max(math.floor( (z - r + (int(param.PeriodicBoundaryCondition) + 1)*param.zmax)/param.pixelsize )    , 0)
    Nz_max = min(math.floor( (z + r + (int(param.PeriodicBoundaryCondition) + 1)*param.zmax)/param.pixelsize ) + 2, param.Nz)
    
    # Create a small binary image with only the considered sphere
    bw_object = np.sqrt( (param.xgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - x)**2 +\
                         (param.ygrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - y)**2 +\
                         (param.zgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - z)**2 ) <= r*param.dil_coeff
                         
    return bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max]



def __RodMask(param,x,y,z,wx,wy,wz,l,r):
    """Return a small binary image of a rod (i.e. spherocylinder) of center (``x``,``y``,``z``), orientation vector (wx,wy,w), length ``l`` and radius
    ``r``, as well as the coordinates [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max] of this image in the whole picture.
    """
    # Compute the smallest box [Nx_min,Nx_max[ x [Ny_min,Ny_max[ x [Nz_min,Nz_max[ entirely containing this spherocylinder. Note the left-hand
    # inclusion and right-hand exclusion to accommodate Python indexing. Grid's index are between 0 and Nx-1 (both included), so the range is cut-off
    # to [0,Nx[.
    Nx_min = max(math.floor( (x - abs(wx)*l/2.0 - r + (int(param.PeriodicBoundaryCondition) + 1)*param.xmax)/param.pixelsize ) - 1, 0)
    Nx_max = min(math.floor( (x + abs(wx)*l/2.0 + r + (int(param.PeriodicBoundaryCondition) + 1)*param.xmax)/param.pixelsize ) + 2, param.Nx)
    
    Ny_min = max(math.floor( (y - abs(wy)*l/2.0 - r + (int(param.PeriodicBoundaryCondition) + 1)*param.ymax)/param.pixelsize ) - 1, 0)
    Ny_max = min(math.floor( (y + abs(wy)*l/2.0 + r + (int(param.PeriodicBoundaryCondition) + 1)*param.ymax)/param.pixelsize ) + 2, param.Ny)
    
    Nz_min = max(math.floor( (z - abs(wz)*l/2.0 - r + (int(param.PeriodicBoundaryCondition) + 1)*param.zmax)/param.pixelsize ) - 1, 0)
    Nz_max = min(math.floor( (z + abs(wz)*l/2.0 + r + (int(param.PeriodicBoundaryCondition) + 1)*param.zmax)/param.pixelsize ) + 2, param.Nz)
    
    # Compute the orthogonal projection of each point of the reduced grid onto the central segment of the spherocylinder
    # For each point of the reduced grid, compute its orthogonal projection onto the central segment of the spherocylinder
    p = (param.xgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - x)*wx +\
        (param.ygrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - y)*wy +\
        (param.zgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - z)*wy
    
    # For each point of the reduced grid, compute the distance between its projection and itself
    d = np.sqrt( (param.xgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - x)**2 +\
                 (param.ygrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - y)**2 +\
                 (param.zgrid[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] - z)**2 - p**2 )
    
    # Create a small binary image with the central cylinder
    bw_object1 = (np.abs(p) <= l/2.0) & (d <= r)
    # Create a small binary image with the two end-point spheres
    bw_object2 = np.sqrt(d**2 + (np.abs(p) - l/2.0)**2) <= r
    # Sum up the two previous images to obtain a spherocylinder
    bw_object = bw_object1 | bw_object2
                         
    return bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max]



def __InsertAllSpheres(param,SpheresSet,bw):
    """Insert in the binary image ``bw`` all the elements of ``SpheresSet`` (and their virtual duplicates if ``PeriodicBoundaryCondition`` is True), 
    with radius multiplied by ``dil_coeff`` to help with the segmentation to come.
    """
    if param.PeriodicBoundaryCondition:
        # Insert each object and its duplicates into the image
        for index in range(param.Nspheres):
            for per in range(8):
                # Compute position of the 'per'-th duplicate of object 'index'
                [xper,yper,zper] = __FindVirtualDuplicate(param,SpheresSet["X"][index],SpheresSet["Y"][index],SpheresSet["Z"][index],per)
                # Retrieve binary mask corresponding to this duplicate, with radius multiplied by 'dil_coeff'
                bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(param,xper,yper,zper,SpheresSet["R"][index]*param.dil_coeff)
                # Insert this duplicate in the image 'bw'
                bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] | bw_object
    else:
        # Insert each object into the image
        for index in range(param.Nspheres):
            # Retrieve binary mask corresponding to object 'index' with radius multiplied by 'dil_coeff'
            bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(param,SpheresSet["X"][index],SpheresSet["Y"][index],\
                                                                                        SpheresSet["Z"][index],SpheresSet["R"][index]*param.dil_coeff)
            # Insert object 'index' in the image 'bw'
            bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] | bw_object



def __SubstractAllRods(param,RodsSet,bw):
    """Substract from the binary image ``bw`` all the elements of ``RodsSet`` (and their virtual duplicates if ``PeriodicBoundaryCondition`` is True).
    """
    if param.PeriodicBoundaryCondition:
        # Substract each object and its duplicates from the image
        for index in range(param.Nrods):
            for per in range(8):
                # Compute position of the 'per'-th duplicate of object 'index'
                [xper,yper,zper] = __FindVirtualDuplicate(param,RodsSet["X"][index],RodsSet["Y"][index],RodsSet["Z"][index],per)
                # Retrieve binary mask corresponding to this duplicate
                bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __RodMask(param,xper,yper,zper,\
                                                                                   RodsSet["wX"][index],RodsSet["wY"][index],RodsSet["wZ"][index],\
                                                                                   RodsSet["L"][index],RodsSet["R"][index])
                # Substract this duplicate from the image 'bw'
                bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] & (~bw_object)
    else:
        # Substract each object from the image
        for index in range(param.Nrods):
            # Retrieve binary mask corresponding to object 'index'
            bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __RodMask(param,RodsSet["X"][index],RodsSet["Y"][index],RodsSet["Z"][index],\
                                                                               RodsSet["wX"][index],RodsSet["wY"][index],RodsSet["wZ"][index],\
                                                                               RodsSet["L"][index],RodsSet["R"][index])
            # Substract object 'index' from the image 'bw'
            bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] & (~bw_object)



def Create3Dbinaryimage(param,SpheresSet,RodsSet):
    """Return a binary image with 1's for spheres and 0's for background.
    
    If rod-like objects are provided, pixels located within a rod will be set to 0 (even if overlapping with a sphere).
    """
    # Create void binary image with the right size
    bw_tot = param.xgrid*0.0 <= -1
    
    # Insert all the elements of SpheresSet into the image
    __InsertAllSpheres(param,SpheresSet,bw_tot)
        
    # Fill closed holes between objects       
    bw_tot = scim.binary_fill_holes(bw_tot)
    
    # Substract all the elements of RodsSet from the image
    if RodsSet != None:
        __SubtractAllRods(param,RodsSet,bw_tot)
        bw_tot = scim.binary_fill_holes(bw_tot)
    
    return bw_tot


# PreprocessAndApplyWatershed
# ApplyPreprocessThenWatershed
# WatershedWithPreprocess
# ApplyWatershedWithPreprocess
def MapRegionsUsingWatershed(bw,pixelsize,smooth_coeff):
    """Return a labeled map of the different regions in the binary image ``bw``, i.e. a numpy.ndarray of the same shape than ``bw`` with 0's for 
    background and different strictly positive integers for each region. Each region corresponds to a cluster of spheres. 
    
    To identify the regions, this function first process the binary image ``bw`` into a grayscale image/ topographic map using the procedure 
    described in
    ****** matlab page ******************
    
    then apply a watershed algorithm to this grayscale image using its local peaks as regions' seeds.
    """
    # For each pixel of the input binary image, compute the Euclidean distance to the nearest zero pixel, using ``pixelsize`` as the length of the 
    # pixels in each direction.
    distance_map = scim.distance_transform_edt(bw,sampling=pixelsize)
    
    # Apply h-minimum transform to smooth the distance map.
    # --- This function was designed to reproduce/mimic the results of Matlab's imhmin function on 3D images.
    distance_map_smoothed = - imhmin(distance_map,smooth_coeff) + 1.0
    
    # Find the local peaks in the smoothed distance map and return a 3D binary image with 1's at the position of local peaks and 0's elsewhere. 
    local_max_map = local_maxima(distance_map_smoothed)
    
    # Label the different peaks using connected component analysis with full connectivity in the three directions. The labeled pixels will serve as
    # seeds for the watershed process.
    seed_for_watershed = scim.label(local_max_map,np.ones((3,3,3)))[0]
    
    # Apply watershed algorithm.
    ClusterMap = watershed(-distance_map_smoothed, seed_for_watershed, mask=bw)
    
    return ClusterMap



def __MergePeriodicCluster(param,SpheresSet,ClusterIndex):
    """Merge the various instances of the clusters extending through the border of the domain and return a filtered array containing the definitive 
    cluster index of each element of SpheresSet.
    
    Only useful is the domain have periodic boundary condition.
    """
    PeriodicClusterIndex = - np.ones(param.Nspheres)
    ClusterRedirection = np.arange(np.max(ClusterIndex))
    SpheresSetTranslatedPosition = np.zeros( (param.Nspheres,3) )
    
    # Identify the real clusters, that is the clusters passing through the central (not duplicated) part of the system
    list_cluster_real, list_cluster_size = np.unique(ClusterIndex[:,0],return_counts=True)
    
    # Sort clusters by their size, in descending order 
    sorting_index = list_cluster_size.argsort()
    list_cluster_real = list_cluster_real[sorting_index[::-1]]
    
    # Iterate over all real clusters
    for c in list_cluster_real:
        
        # For each object...
        for index in range(param.Nspheres):
            # ...still not definitely attributed to a cluster...
            if PeriodicClusterIndex[index] == -1:
                # ...check if one of its duplicates...
                for per in range(8):
                    # ...pertain to cluster number c
                    if ClusterIndex[index,per] == c:
                        # If True, attribute this object to c or whatever other cluster c redirect to
                        PeriodicClusterIndex[index] = ClusterRedirection[c]
                        # Save the position of the duplicate that made the connection
                        SpheresSetTranslatedPosition[index,:] = __FindVirtualDuplicate(param,SpheresSet["X"][index],SpheresSet["Y"][index],\
                                                                                       SpheresSet["Z"][index],per)
                        # Note that the cluster this object was originally attributed to in fact redirect to c
                        ClusterRedirection[ ClusterIndex[index,0] ] = ClusterRedirection[c]
                        # Once connection has been done, no need to check the other duplicates
                        break
    
    return PeriodicClusterIndex, SpheresSetTranslatedPosition



def __SimplifyCluster(param,ClusterIndex):  
    """Suppress the clusters smaller (in term of number of objects) than ``MinNumberOfObjectInCluster`` 
    
    Also simplify cluster indexing by renumbering the remaining clusters continuously starting from 0. Note that non-attributed objects are denoted 
    by -1.
    """
    # Initialize number of valid clusters (N.B : cluster indexing starts from 0)
    Ncluster = -1
    # Initialize attribution of objects to cluster : non-attributed objects are denoted by -1
    SimplifiedClusterIndex = - np.ones(param.Nspheres)
    # Identify existing clusters and compute the number of objects in each of them
    list_cluster, list_cluster_size = np.unique(ClusterIndex,return_counts=True)
    
    # Iterate over all existing clusters
    for index in range(len(list_cluster)):
        # Check if the number of objects in this cluster is greater or equal to threshold
        if list_cluster_size[index] >= param.MinNumberOfObjectInCluster:
            # Increment number of valid clusters
            Ncluster += 1
            # Renumber all objects in this cluster (N.B : cluster numbering start from 0)
            SimplifiedClusterIndex = np.where(ClusterIndex==list_cluster[index], Ncluster, SimplifiedClusterIndex)
    
    return SimplifiedClusterIndex



def ClusterizeFromMap(param,SpheresSet,ClusterMap):
    ## """Add to the original ``SpheresSet`` DataFrame a column labeled ``header`` and containing the cluster index attributed to each sphere
    ##    according to the ``ClusterMap``, with clusters continuously numbered starting from 0 and non-attributed objects denoted by -1.
    ## """
    """Return an array referencing the cluster index attributed to each element of ``SpheresSet`` according to the ``ClusterMap``, with clusters 
    continuously numbered starting from 0 and non-attributed objects denoted by -1.
    """
    if param.PeriodicBoundaryCondition:
        # Create array in which to store the cluster index of each object and its duplicates
        DuplicatedClusterIndex = np.zeros((param.Nspheres,8))
                                          
        # Retrieve cluster index for each object and its duplicates
        for index in range(param.Nspheres):
            for per in range(8):
                # Compute position of the 'per'-th duplicate of object 'index'
                [xper,yper,zper] = __FindVirtualDuplicate(param,SpheresSet["X"][index],SpheresSet["Y"][index],SpheresSet["Z"][index],per)
                # Retrieve object mask
                object_mask, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(param,xper,yper,zper,SpheresSet["R"][index])
                # Using this mask, retrieve watershed information on the object
                object_seg = ClusterMap[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max]
                object_seg = object_seg[object_mask]
                # Find the most common value attributed to the pixels of the considered object by the watershed segmentation.
                # Since watershed number clusters starting from 1 (with background as 0) while Python indexing start from 0, shift numbering by -1 so
                # that clusters are now numbered starting from 0, with -1 as background.
                DuplicatedClusterIndex[index,per] = np.argmax(np.bincount(object_seg)) - 1
        
        # Merge the various instances of the clusters extending through the border of the domain
        ClusterIndex, SpheresSetTranslatedPosition = __MergePeriodicCluster(param,SpheresSet,DuplicatedClusterIndex)
                                                
    else:
        # Create array in which to store the cluster index of each object
        ClusterIndex = np.zeros(param.Nspheres)
    
        # Retrieve cluster index for each object
        for index in range(param.Nspheres):
            # Retrieve object mask
            object_mask, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(param,SpheresSet["X"][index],SpheresSet["Y"][index],\
                                                                                          SpheresSet["Z"][index],SpheresSet["R"][index])
            # Using this mask, retrieve watershed information on the object
            object_seg = ClusterMap[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max]
            object_seg = object_seg[object_mask]
            # Find the most common value attributed to the pixels of object i by the watershed segmentation. Since watershed number clusters starting
            # from 1 (with background as 0) while Python indexing start from 0, shift numbering by -1 so that clusters are now numbered starting from
            # 0, with -1 as background.
            ClusterIndex[index] = np.argmax(np.bincount(object_seg)) - 1
    
    
    # Renumber clusters continuously starting from 0, while suppressing too small clusters
    ClusterIndex = __SimplifyCluster(param,ClusterIndex)
    
    return ClusterIndex