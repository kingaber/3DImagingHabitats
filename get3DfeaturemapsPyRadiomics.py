# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:25:05 2020

@author: kbernatowicz
"""
import os
import SimpleITK as sitk
from radiomics import featureextractor
import matplotlib.pyplot as plt
from scipy import ndimage

def process_image(idir, sets, flag_plot=False, flag_retest=False):
    odir = f"{idir}_featuremap_{sets}"
    if not os.path.exists(odir):
        os.makedirs(odir)

    for root, dirs, files in os.walk(idir):
        for file in files:
            try:
                if flag_retest:
                    if not file.endswith("-retest-img.nrrd"):
                        continue
                    patid = file.split("-retest-img.nrrd")[0]
                else:
                    if not file.endswith("-test-img.nrrd"):
                        continue
                    patid = file.split("-test-img.nrrd")[0]

                image_file = os.path.join(idir, file)
                mask_file = os.path.join(idir, f"{patid}-{'retest-' if flag_retest else ''}seg.nrrd")
                eroded_file = os.path.join(odir, f"Cropped_{patid}-{'retest-' if flag_retest else ''}img.nrrd")

                print("-----------------")
                print("Processing image:", image_file)
                print("Processing mask:", mask_file)

                mask = sitk.ReadImage(mask_file)
                mask_arr = sitk.GetArrayFromImage(mask)
                erosion = ndimage.binary_erosion(mask_arr, iterations=2)
                erosion_mask = sitk.GetImageFromArray(erosion.astype(mask_arr.dtype))
                erosion_mask.CopyInformation(mask)
                sitk.WriteImage(erosion_mask, eroded_file)

                params = os.path.join(idir, f"{sets}.yaml")
                extractor = featureextractor.RadiomicsFeatureExtractor(params)
                voxel_result = extractor.execute(image_file, eroded_file, voxelBased=True)

                for key, val in voxel_result.items():
                    if isinstance(val, sitk.Image):
                        if flag_plot:
                            parameter_map = sitk.GetArrayFromImage(val)
                            plt.figure()
                            plt.imshow(parameter_map[int(parameter_map.shape[0] / 2), :, :])
                            plt.title(key)

                        featuremap_file = os.path.join(odir, f"{'_'.join([patid, 'test', key]) if flag_retest else '_'.join([patid,'retest', key])}.nrrd")
                        sitk.WriteImage(val, featuremap_file)
                    else:
                        # Diagnostic feature
                        print(f"{key}: {val}")

                if not voxel_result:
                    print("No features extracted!")
            except Exception as e:
                print("An error occurred:", str(e))

if __name__ == "__main__":
    idir = r"..."  # Input directory e.g. "C:\Users\user1"
    settings = [""]  # Can be a list. Used for PyRadiomics settings, feature extraction parameter combination e.g. "R1B12"
    flag_plot = False  # Use True to visualize feature maps
    flag_retest = False  # Use True if retest image has to be analyzed

    print("*************************************")
    print("Compute 3D features using PyRadiomics extractor")
    print("*************************************")

    for sets in settings:
        process_image(idir, sets, flag_plot, flag_retest)
