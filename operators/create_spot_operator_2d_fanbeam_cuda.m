function [ W ] = create_spot_operator_2d_fanbeam_cuda(n, SOD, ODD, detPixelSize, nDetectors, angles)
%CREATE_SPOT_OPERATOR_2D_FANBEAM_CUDA Create ASTRA Spot operator (CUDA GPU)
%   W = create_spot_operator_2d_fanbeam_cuda(n, SOD, ODD, detPixelSize, 
%   nDetectors, angles) produces an operator which can be used like the
%   system matrix A of a tomographic forward model in 2D fan beam geometry 
%   using a flat detector and a GPU-based projector. The parameters 
%   specifying the imaging geometry are:
%   n:                  The reconstruction grid edge length in pixels.
%   SOD:                The source-to-origin distance in mm (the origin is
%                       defined as the center of rotation).
%   ODD:                The origin-to-detector distance in mm.
%   detPixelSize:       The physical size of the detector pixels in mm.
%   nDetectors:         The number of detectors in a single projection
%                       (i.e. one sinogram row).
%   angles:             A vector containing the projection angles in
%                       degrees.
%
%   NOTE: Using this function requires installed versions of the ASTRA 
%   Tomography Toolbox and the Spot Operator Toolbox.
%
%   Based on the codes by
%   Alexander Meaney, University of Helsinki
%   Created:            29.5.2018
%   Last edited:        2.6.2025 by Tommi Heikkilä, LUT
%   Also see HelTomo toolbox: https://github.com/Diagonalizable/HelTomo

% Specify reconstruction volume geometry
volGeom = astra_create_vol_geom(n, n);

% Specify projection geometry.

% Geometric magnification and effective pixel size in reconstruction
M = (SOD + ODD) / SOD;
effPixelSize = detPixelSize / M;

% Convert to radians
anglesRad = angles .* (pi/180);

% In the projection geometry, all units of distance are given as multiples
% of effPixelSize
projGeom = astra_create_proj_geom('fanflat', ...          % Geometry
                                  M, ...                  % Detector pixel width
                                  nDetectors, ...         % Number of detector pixels 
                                  anglesRad, ...          % Angles in radians
                                  SOD / effPixelSize, ... % Source-origin distance
                                  ODD / effPixelSize);    % Origin-detector distance

% Create Spot operator using GPU
W = opTomo('cuda', projGeom, volGeom);
end

