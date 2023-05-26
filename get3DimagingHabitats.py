# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:26:52 2021

@author: kbernatowicz
"""
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

def compute_habitats(evaluate_dir, patientid):
    odir = evaluate_dir + '_habitats'
    if not os.path.exists(odir):
        os.makedirs(odir)

    data_df = pd.DataFrame()

    for p in patientid:
        db_test = pd.DataFrame()
        db_retest = pd.DataFrame()

        for root, dirs, files in os.walk(evaluate_dir):
            for file in files:
                if p+'_test_original' in file:
                    feature = file.split('_test_original_')[1].split('.nrrd')[0]
                    if not feature == 'glcm_MCC': # feature to be exluded from analysis
                        test_file = os.path.join(evaluate_dir, file)
                        test_raw = sitk.ReadImage(test_file)
                        test_arr = sitk.GetArrayFromImage(test_raw)

                        retest_str = p+'_retest_original'+file.split('_test_original')[1]
                        retest_file = os.path.join(evaluate_dir, retest_str)
                        retest_raw = sitk.ReadImage(retest_file)
                        retest_arr = sitk.GetArrayFromImage(retest_raw)

                        X = test_arr.flatten()
                        robu_scaler = MinMaxScaler().fit(X.reshape(-1, 1))
                        X_scaled = robu_scaler.transform(X.reshape(-1, 1))
                        X_arr = X_scaled.reshape(np.shape(test_arr))

                        X2 = retest_arr.flatten()
                        X_scaled2 = robu_scaler.transform(X2.reshape(-1, 1))
                        X_arr2 = X_scaled2.reshape(np.shape(retest_arr))

                        if feature == 'glcm_Autocorrelation': #use this feature as a mask
                            nanrm_test = X_arr
                            nanrm_retest = X_arr2
                            test_Nanrow = np.isnan(nanrm_test)
                            retest_Nanrow = np.isnan(nanrm_retest)

                        test = X_arr[~test_Nanrow]
                        retest = X_arr2[~retest_Nanrow]

                        df_test = pd.DataFrame(test, columns=[feature])
                        db_test = pd.concat([db_test, df_test], axis=1)

                        df_retest = pd.DataFrame(retest, columns=[feature])
                        db_retest = pd.concat([db_retest, df_retest], axis=1)

        X = db_test.replace(np.nan, 0)
        pca = PCA(n_components=X.shape[1])
        filtered_test_pca = pca.fit_transform(X)

        evar = np.sum(pca.explained_variance_ratio_) * 100
        print("Explained Variance: %s" % evar)

        X_r = db_retest.replace(np.nan, 0)
        filtered_retest_pca = pca.transform(X_r)

        Knum = determine_optimal_clusters(filtered_test_pca)

        habFileName = os.path.join(odir, f'{p}_K{Knum}_test.nrrd')
        kmeans = KMeans(n_clusters=Knum, init='k-means++', random_state=0)
        y_kmeans = kmeans.fit_predict(filtered_test_pca)
        cluster_arr_test = apply_cluster_labels(test_arr, test_Nanrow, y_kmeans)
        save_cluster_image(cluster_arr_test, test_raw, habFileName)

        habFileName = os.path.join(odir, f'{p}_K{Knum}_retest.nrrd')
        y_kmeans_retest = kmeans.predict(filtered_retest_pca)
        cluster_arr_retest = apply_cluster_labels(retest_arr, retest_Nanrow, y_kmeans_retest)
        save_cluster_image(cluster_arr_retest, retest_raw, habFileName)

        dice_results = compute_dice_similarity(cluster_arr_test, cluster_arr_retest)
        data_df = data_df.append(dice_results, ignore_index=True)

    plot_dice_similarity(data_df)

def determine_optimal_clusters(X):
    # Code for determining the optimal number of clusters
    # Uncomment and modify this section if needed
    # distortion = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    #     kmeans.fit(X)
    #     distortion.append(kmeans.inertia_)
    #
    # plt.plot(range(1, 11), distortion)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()

    # Hardcoded optimal number of clusters
    Knum = 4
    return Knum

def apply_cluster_labels(arr, Nanrow, labels):
    cluster_arr = np.empty(np.shape(arr))
    cluster_arr[:] = np.nan
    cluster_arr[~Nanrow] = labels
    return cluster_arr

def save_cluster_image(cluster_arr, raw, filename):
    cluster_img = sitk.GetImageFromArray(cluster_arr)
    cluster_img.CopyInformation(raw)
    sitk.WriteImage(cluster_img, filename)

def compute_dice_similarity(cluster_arr_test, cluster_arr_retest):
    test_clusters = np.unique(cluster_arr_test)
    retest_clusters = np.unique(cluster_arr_retest)

    dice_results = {'Test Cluster': [], 'Retest Cluster': [], 'Dice Similarity Coefficient': []}

    for test_cluster in test_clusters:
        if test_cluster != -1:
            test_mask = cluster_arr_test == test_cluster
            test_volume = np.sum(test_mask)

            max_dice = 0
            max_retest_cluster = -1

            for retest_cluster in retest_clusters:
                if retest_cluster != -1:
                    retest_mask = cluster_arr_retest == retest_cluster
                    retest_volume = np.sum(retest_mask)

                    overlap = np.logical_and(test_mask, retest_mask)
                    overlap_volume = np.sum(overlap)

                    dice = 2 * overlap_volume / (test_volume + retest_volume)

                    if dice > max_dice:
                        max_dice = dice
                        max_retest_cluster = retest_cluster

            dice_results['Test Cluster'].append(test_cluster)
            dice_results['Retest Cluster'].append(max_retest_cluster)
            dice_results['Dice Similarity Coefficient'].append(max_dice)

    return dice_results

def plot_dice_similarity(data_df):
    sns.boxplot(data=data_df, x='Test Cluster', y='Dice Similarity Coefficient')
    plt.title('Dice Similarity Coefficient')
    plt.show()

# Example usage
evaluate_dir = 'path/to/evaluate/dir'
patientid = ['patient1', 'patient2', 'patient3']

compute_habitats(evaluate_dir, patientid)
