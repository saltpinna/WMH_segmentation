import subprocess
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from natsort import natsorted
import os

def preprocessing(t1_image_path, flair_image_path, output_path):
    """
    Input data should be organized in a folder named with the case identifier, containing the T1 and FLAIR images
    """
    
    ### Runs FSL-FLIRT (co-registration of T1 to FLAIR) as a subprocess in Python ###
    ref_file = flair_image_path    #FLAIR
    input_file = t1_image_path   #T1
    if not os.path.isdir(output_path + 'coregistered'):
        os.mkdir(output_path + 'coregistered')
    output_file = output_path + 'coregistered/' + ref_file.split('/')[-2] + '_0000.nii.gz'
    command = ['/home/neuro_nph/fsl/bin/flirt', '-in', input_file, '-ref', ref_file, '-out', output_file]
    subprocess.run(command)
    
    
    ### Skull-stripping ###
    input_file_path = output_path + 'coregistered'
    if not os.path.isdir(output_path + 'brain_masks'):
        os.mkdir(output_path + 'brain_masks')
    mask_output_path = output_path + 'brain_masks'

    # Runs the nnU-Net for skullstripping
    command = ['nnUNet_predict', '-i', input_file_path, '-o', mask_output_path, '-tr', 'nnUNetTrainerV2_150epochs', '-ctr', 'nnUNetTrainerV2CascadeFullRes', '-m', '3d_fullres', '-p', 'nnUNetPlansv2.1', '-t', 'Task500_Skullstripping']
    subprocess.run(command)
    
    #Loads images and brain mask
    t1_image_file = nib.load(output_path + 'coregistered/' + ref_file.split('/')[-2] + '_0000.nii.gz')
    t1_image_data = t1_image_file.get_fdata()
    flair_image_file = nib.load(flair_image_path)
    flair_image_data = flair_image_file.get_fdata()
    mask_file = nib.load(output_path + 'brain_masks' + '/' + ref_file.split('/')[-2] + '.nii.gz')
    mask_data = mask_file.get_fdata()

    # Applying the brain mask to remove non-brain tissue and save skullstripped T1 and FLAIR images
    t1_brain_image = t1_image_data*mask_data   
    flair_brain_image = flair_image_data*mask_data
    skullstripped_t1_image = nib.Nifti1Image(t1_brain_image, flair_image_file.affine, flair_image_file.header)
    skullstripped_flair_image = nib.Nifti1Image(flair_brain_image, flair_image_file.affine, flair_image_file.header)
    if not os.path.isdir(output_path + 'skullstripped'):
        os.mkdir(output_path + 'skullstripped')
    skullstripped_path = output_path + 'skullstripped/'
    nib.save(skullstripped_t1_image, skullstripped_path + ref_file.split('/')[-2] + '_0000.nii.gz')
    nib.save(skullstripped_flair_image, skullstripped_path + ref_file.split('/')[-2] + '_0001.nii.gz')
    
    
    ### Bias field correction with N4 filter ###
    t1_image = sitk.ReadImage(output_path + 'skullstripped/' + ref_file.split('/')[-2] + '_0000.nii.gz', sitk.sitkFloat32)
    flair_image = sitk.ReadImage(output_path + 'skullstripped/' + ref_file.split('/')[-2] + '_0001.nii.gz', sitk.sitkFloat32)
    corrected_t1 = sitk.N4BiasFieldCorrection(t1_image)
    corrected_flair = sitk.N4BiasFieldCorrection(flair_image)


    ### Saving preprocessed images to specified output location ###
    if not os.path.isdir(output_path + 'final_preprocessed_images'):
        os.mkdir(output_path + 'final_preprocessed_images')
    sitk.WriteImage(corrected_t1, output_path + 'final_preprocessed_images/WMH_' + ref_file.split('/')[-2] + '_0000.nii.gz')
    sitk.WriteImage(corrected_flair, output_path + 'final_preprocessed_images/WMH_' + ref_file.split('/')[-2] + '_0001.nii.gz')

    
if __name__ == '__main__':
    path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/test_before_preprocessing/'
    output_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/preprocessed_test_set/'
    dir_list = natsorted(os.listdir(path))
    for dir in dir_list:
        t1_image_path = path + dir +'/3DT1.nii.gz'
        flair_image_path = path + dir +'/FLAIR.nii.gz'
        preprocessing(t1_image_path, flair_image_path, output_path)