# 3DImagingHabitats (last updated: 2023-05-26)

This repository contains python codes related to the imaging habitats article (Bernatowicz et al 2021). It can be used to:
- Extract 3D feature maps using PyRadiomics uding **get3DfeaturemapsPyRadiomics.py**
-	Cluster selected 3D features to create 3D “imaging habitat maps” Bernatowicz et al 2021 Sci.Rep. using **get3DimagingHabitats.py**


**To run the codes ensure you have the necessary prerequisites:**
-	Python installed on your system (version 3.6 or above).
-	Required libraries: SimpleITK, radiomics, matplotlib, and scipy. You can install them using pip:
`pip install radiomics pandas numpy SimpleITK matplotlib seaborn scikit-learn`

**get3DfeaturemapsPyRadiomics.py**
1.	Update the code to match your specific requirements:
  -	Set the idir variable to the path of the input directory containing the image and mask files. 
  -	Modify the settings list if needed, specifying the PyRadiomics settings or feature extraction parameter combinations.
  -	Adjust the values of flag_plot and flag_retest as per your requirements.
2.	Open a terminal or command prompt and navigate to the directory where you saved the Python file.
3.	Run the code by executing the following command: `python get3DfeaturemapsPyRadiomics.py`
4.	The resulting feature maps will be saved in a new directory named <input_directory>_featuremap_<settings>, where <input_directory> represents the input directory path, and <settings> corresponds to the value in the settings list.

**get3DimagingHabitats.py**
1. Prepare the input data:
  - Ensure that you have the necessary input data in the expected format.
  - Create a directory that contains the image files for evaluation. Each image file should follow the naming convention: <patientid>_original_<feature>.nrrd.
  - Update the evaluate_dir variable in the code with the path to this directory.
  - Define the patientid list with the relevant patient IDs.
2. Run the code:
  - Open a terminal or command prompt.
  - Navigate to the directory where you saved the Python script.
  - Run the script by executing the following command: `python get3DfeaturemapsPyRadiomics.py`
3.	Review the results:
  - Once the code finishes running, you can check the output directory specified by evaluate_dir + '_habitats'.
  - Inside this directory, you'll find the habitat images for each patient, labeled with the number of clusters.
  - The box plot of the Dice similarity coefficients will be displayed on your screen.

**Please cite this paper if you found this repository useful:**
Bernatowicz, K., Grussu, F., Ligero, M. et al. Robust imaging habitat computation using voxel-wise radiomics features. Sci Rep 11, 20133 (2021). https://doi.org/10.1038/s41598-021-99701-2
![image](https://github.com/kingaber/3DImagingHabitats/assets/58729619/93059b76-a3bd-449c-ba99-ae21bb469519)
