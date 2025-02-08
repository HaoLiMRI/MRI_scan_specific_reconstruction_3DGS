# MRI_scan_specific_reconstruction_3DGS

Using original scheme of 3DGS to directly predict the intensity, initial phase, and 3D Gaussian distribution of each voxel.
The intensity and initial phase will be encoded based on the frequency and phase-encoding theory to adjust the phase, and the sum of intensity in complex form from all voxels will be filled in the coresponding element of K-space for each coil element.
The loss will be calculated between the reconstructed K-space and the acquired undersampled K-space to update the predicted parameters. 
