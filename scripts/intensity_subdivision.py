import subprocess
import os
import nibabel as nib
import numpy as np

def intensity_subdivision(t1_image_path, output_path):
    ### Runs FSL-FAST tissue segmentation as a subprocess in Python
    input_file = t1_image_path   # T1
    case = input_file.split('/')[-1].split('.')[0].split('_0000')[0]
    if not os.path.isdir(output_path + case):
        os.mkdir(output_path + case)
    os.chdir(output_path + case)
    command = ['/home/neuro_nph/fsl/bin/fast', '-t', '1', '-n', '3', '-H', '0.1', '-I', '4', '-o', './', input_file]
    subprocess.run(command)

    ### Loads the segmentation mask for gray matter, white matter and cerebrospinal fluid
    seg_im = nib.load(output_path + case + '/_pveseg.nii.gz')   #1: gray matter, 2: white matter, 3: cerebrospical fluid
    seg_im = seg_im.get_fdata()

    ### Loads WMH mask image with spatial subdivision
    spatial_subdivision_masks = os.listdir(output_path + '/../spatial_subdivision/')
    wmh_im = nib.load(output_path + '/../spatial_subdivision/' + spatial_subdivision_masks[0])
    wmh_mask = wmh_im.get_fdata()

    ### Classify voxels in wmh mask based on class in segmentation mask
    wmh_class = np.zeros_like(wmh_mask)
    #1: non T1-hypointense DWMHs
    wmh_class[(wmh_mask == 1) & (seg_im == 2)] = 1
    #2: T1-hypointense DWMHs
    wmh_class[(wmh_mask == 1) & ((seg_im == 1) | (seg_im == 3))] = 2 
    #3: non T1-hypointense PWMHs
    wmh_class[(wmh_mask == 2) & (seg_im == 2)] = 3 
    #4: T1-hypointense PWMHs
    wmh_class[(wmh_mask == 2) & ((seg_im == 1) | (seg_im == 3))] = 4 
    #5: non T1-hypointense JVWMHs
    wmh_class[(wmh_mask == 3) & (seg_im == 2)] = 5 
    #6: non T1-hypointense JVWMHs
    wmh_class[(wmh_mask == 3) & ((seg_im == 1) | (seg_im == 3))] = 6
    #7: non T1-hypointense JCWMHs
    wmh_class[(wmh_mask == 4) & (seg_im == 2)] = 7
    #8: non T1-hypointense JCWMHs
    wmh_class[(wmh_mask == 4) & ((seg_im == 1) | (seg_im == 3))] = 8
    intensity_divided_mask = nib.Nifti1Image(wmh_class, wmh_im.affine)
    nib.save(intensity_divided_mask, output_path + case + '/intensity_subdivision.nii.gz')
    

if __name__ == '__main__':
    input_image_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/intensity_subdivision/WMH_3_0000.nii.gz'
    output_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/intensity_subdivision/'
    intensity_subdivision(input_image_path, output_path)