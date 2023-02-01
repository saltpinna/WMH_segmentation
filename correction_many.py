import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib

# Bias field corrected using N4
path = r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data/training/Amsterdam/GE3T/"

# Loops through training set (Amsterdam in this case) and loads FLAIR + mask image, performs N4 correction and saves resulting NIFTI image
# for i in range(100, 145):
#     sitk_img = sitk_img = sitk.ReadImage(path + str(i) + '/orig/FLAIR.nii.gz')
#
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     numberFittingLevels = 4
#
#     sitk_img_new = sitk.Cast(sitk_img, sitk.sitkFloat32)
#
#     corrected_img = corrector.Execute(sitk_img_new)
#     log_bias_field = corrector.GetLogBiasFieldAsImage(sitk.Cast(sitk_img_new, sitk.sitkUInt16))
#     corrected_img_full_res = sitk_img_new/sitk.Exp(log_bias_field)
#
#     # For saving the corrected image as a Nifti file
#     sitk.WriteImage(corrected_img_full_res, r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data/corrected/corrected" + str(i) + '.nii.gz', imageIO='NiftiImageIO')

# For plotting
# corrected_img_array = sitk.GetArrayFromImage(corrected_img_full_res)
# orig_img = sitk.GetArrayFromImage(sitk_img)
fig = plt.figure()
subplot_nr = 1
for i in range(100, 104):
    corrected_img = nib.load(r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data/testing/corrected" + str(i) + '.nii.gz').get_fdata()
    orig_img = nib.load(r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data/training/Amsterdam/GE3T/" + str(i) + '/orig/FLAIR.nii.gz').get_fdata()
    ax1 = fig.add_subplot(2, 4, subplot_nr)
    ax1.imshow(corrected_img[66, :, :], cmap='gray', origin='lower')
    ax1.title.set_text('Corrected')
    ax2 = fig.add_subplot(2, 4, subplot_nr + 1)
    ax2.imshow(orig_img[66, :, :], cmap='gray', origin='lower')
    ax2.title.set_text('Original')
    subplot_nr += 2
plt.show()
