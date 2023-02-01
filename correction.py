import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

path = r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data"
# img_path = nib.load(path + '/training/Amsterdam/GE3T/100/orig/FLAIR.nii.gz')
# mask_path = nib.load(path + '/training/Amsterdam/GE3T/100/wmh.nii.gz')

# Bias field corrected using N4, Cast used to get right datat type for input to functions
sitk_img = sitk.ReadImage(path+'/training/Amsterdam/GE3T/100/orig/FLAIR.nii.gz')

corrector = sitk.N4BiasFieldCorrectionImageFilter()
numberFittingLevels = 4

sitk_img_new = sitk.Cast(sitk_img, sitk.sitkFloat32)

corrected_img = corrector.Execute(sitk_img_new)
log_bias_field = corrector.GetLogBiasFieldAsImage(sitk.Cast(sitk_img_new, sitk.sitkUInt16))
corrected_img_full_res = sitk_img_new/sitk.Exp(log_bias_field)

# For saving the corrected image as a Nifti file
# sitk.WriteImage(corrected_img_full_res, path + '/corrected/corrected.nii.gz', imageIO='NiftiImageIO')

# For plotting
corrected_img_array = sitk.GetArrayFromImage(corrected_img_full_res)
orig_img = sitk.GetArrayFromImage(sitk_img)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.imshow(corrected_img_array[50, :, :], cmap='gray', origin='lower')
ax1.title.set_text('Corrected')
ax2 = fig.add_subplot(2, 1, 2)
ax2.imshow(orig_img[50, :, :], cmap='gray', origin='lower')
ax2.title.set_text('Original')
plt.show()
