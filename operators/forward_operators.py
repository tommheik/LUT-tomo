import math
import numpy as np
import astra
import matplotlib.pyplot as plt

def createASTRAoperator_fanbeam(N, SOD, ODD, pixelSize, N_det, theta, use_cuda = True):
    """
    # Function for building the OpTomo forward operator

    N         : Reconstruction resolution (N x N image)
    SOD       : Source-to-Origin Distance
    ODD       : Origin-to-Detector Distance
    pixelSize : Width of the detector element after binning
    N_det     : Number of detector elements
    theta     : Projection angles (in degrees)
    use_cuda  : Enable CUDA computations (requires Nvidia GPU) if possible

    T H  --  LUT University
    """

    M = (SOD + ODD) / SOD # Geometric magnification
    effPixelSize = pixelSize / M # Effective pixelSize
    anglesRad = theta*(math.pi/180)
   
    projGeom = astra.create_proj_geom('fanflat', pixelSize, N_det, anglesRad, SOD/effPixelSize, ODD/effPixelSize)
    volGeom = astra.create_vol_geom(N,N)

    if use_cuda and astra.use_cuda():
        projId = astra.create_projector('cuda', projGeom, volGeom)
    else:
        projId = astra.create_projector('strip_fanflat', projGeom, volGeom)
   
    return astra.OpTomo(projId)

def doASTRAfbp_fanbeam(sinogram, N, SOD, ODD, pixelSize, N_det, theta, use_cuda = True):
    """
    # Function for performing filtered back-projection (FBP)

    sinogram  : Data formed into sinogram
    N         : Reconstruction resolution (N x N image)
    SOD       : Source-to-Origin Distance
    ODD       : Origin-to-Detector Distance
    pixelSize : Width of the detector element after binning
    N_det     : Number of detector elements
    theta     : Projection angles (in degrees)
    use_cuda  : Enable CUDA computations (requires Nvidia GPU) if possible

    T H  --  LUT University
    """

    M = (SOD + ODD) / SOD # Geometric magnification
    effPixelSize = pixelSize / M # Effective pixelSize
    anglesRad = theta*(math.pi/180)
   
    projGeom = astra.create_proj_geom('fanflat', pixelSize, N_det, anglesRad, SOD/effPixelSize, ODD/effPixelSize)
    volGeom = astra.create_vol_geom(N,N)
    if use_cuda and astra.use_cuda():
        projId = astra.create_projector('cuda', projGeom, volGeom)
    else:
        projId = astra.create_projector('strip_fanflat', projGeom, volGeom)

    recnId = astra.data2d.create('-vol', volGeom)
    dataId = astra.data2d.link('-sino',projGeom, sinogram)
    # Config
    if use_cuda and astra.use_cuda():
        cfg = astra.astra_dict('FBP_CUDA')
    else:
        cfg = astra.astra_dict('FBP')
    cfg['ProjectorId'] = projId
    cfg['ReconstructionDataId'] = recnId
    cfg['ProjectionDataId'] = dataId
    cfg['option'] = { 'FilterType': 'Ram-Lak' }

    algId = astra.algorithm.create(cfg)
    astra.algorithm.run(algId)
   
    return astra.data2d.get(recnId)

def createASTRAmatrix_fanbeam(N, SOD, ODD, pixelSize, N_det, theta):
    """
    # Function for building the forward operator matrix

    N         : Reconstruction resolution (N x N image)
    SOD       : Source-to-Origin Distance
    ODD       : Origin-to-Detector Distance
    pixelSize : Width of the detector element after binning
    N_det     : Number of detector elements
    theta     : Projection angles (in degrees)

    T H  --  LUT University
    """

    M = (SOD + ODD) / SOD # Geometric magnification
    effPixelSize = pixelSize / M # Effective pixelSize
    anglesRad = theta*(math.pi/180)
   
    projGeom = astra.create_proj_geom('fanflat', pixelSize, N_det, anglesRad, SOD/effPixelSize, ODD/effPixelSize)
    volGeom = astra.create_vol_geom(N,N)
    projId = astra.create_projector('strip_fanflat', projGeom, volGeom)
    matrixId = astra.projector.matrix(projId)
    astra.projector.delete(projId)

    return astra.matrix.get(matrixId)