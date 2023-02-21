# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# stats, histograms...

# import numpy as np
# from pmlb import fetch_data
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

params = {'figure.dpi': 600,
          'font.sans-serif': 'Calibri',
          'font.family': 'sans-serif',
          'axes.titlesize': 11,
          'font.size': 11,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'xtick.minor.size': 0,
          'axes.labelsize': 11,
          'legend.fontsize': 11,
          'legend.handlelength': 1,
          'lines.linewidth': 3,
          'lines.markersize': 8}
matplotlib.rcParams.update(params)

# basic = pd.read_csv('files/BasicEnsemble.csv', header=None)
# cluster = pd.read_csv('files/ClassyEnsemble.csv', header=None)
# lexi = pd.read_csv('files/Lexigarden.csv', header=None)
# classy = pd.read_csv('files/ClassyEnsemble.csv', header=None)
basic = np.genfromtxt('files/BasicEnsemble.csv', delimiter=',')
cluster = np.genfromtxt('files/ClusterEnsemble.csv', delimiter=',')
lexi = np.genfromtxt('files/Lexigarden.csv', delimiter=',')
classy = np.genfromtxt('files/ClassyEnsemble.csv', delimiter=',')

# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
cmap_name = 'tab20'
n_bins = 4
cm = plt.get_cmap(cmap_name)
colors = [cm(i/n_bins) for i in range(4)]

for i in range(3):
    ax = plt.subplot()
    ax.set_title('Samples' if i==0 else 'Features' if i==1 else 'Classes')
    ax.set_xlabel(('samples' if i==0 else 'features' if i==1 else 'classes'))
    l = [basic[:, 2+i], cluster[:, 2+i], lexi[:, 2+i], classy[:, 2+i]]
    ax.hist(l, n_bins, density=False, histtype='bar', align='mid', color=colors,
            label=['Order', 'Cluster', 'Lexigarden', 'Classy'])
    ax.legend(prop={'size': 10})
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    # ax.set_yticks([])
    plt.show()


'''
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

datasets = ['analcatdata_dmft', 'cleveland_nominal', 'flags', 'analcatdata_germangss', 'parity5', 'cleveland', 'backache', 'wine_quality_red', 'wine_quality_white', 'allhyper', 'appendicitis', 'yeast', 'schizo', 'GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1', 'contraceptive', 'GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1', 'cmc', 'coil2000', 'profb', 'tae', 'hepatitis', 'solar_flare_2', 'glass', 'GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1', 'cloud', 'analcatdata_boxing2', 'bupa', 'analcatdata_cyyoung8092', 'calendarDOW', 'prnn_fglass', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001', 'breast_cancer', 'postoperative_patient_data', 'cleve', 'saheart', 'dis', 'spect', 'solar_flare_1', 'Hill_Valley_without_noise', 'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001', 'analcatdata_happiness', 'analcatdata_fraud', 'Hill_Valley_with_noise', 'flare', 'german', 'credit_g', 'haberman', 'diabetes', 'led7', 'heart_h', 'analcatdata_japansolvent', 'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1', 'analcatdata_boxing1', 'ecoli', 'led24', 'molecular_biology_promoters', 'pima', 'horse_colic', 'mfeat_morphological', 'colic', 'analcatdata_asbestos', 'analcatdata_cyyoung9302', 'allrep', 'lupus', 'lymphography', 'hungarian', 'movement_libras', 'sonar', 'confidence', 'adult', 'analcatdata_aids', 'heart_c', 'heart_statlog', 'glass2', 'analcatdata_lawsuit', 'mfeat_zernike', 'vehicle', 'balance_scale', 'biomed', 'auto', 'hayes_roth', 'mfeat_fourier', 'magic', 'page_blocks', 'waveform_40', 'ionosphere', 'shuttle', 'prnn_synth', 'churn', 'crx', 'allbp', 'waveform_21', 'buggyCrx', 'australian', 'phoneme', 'credit_a', 'satimage', 'spectf', 'tokyo1', 'breast_w', 'vowel', 'penguins', 'allhypo', 'labor', 'wdbc', 'hypothyroid', 'monk2', 'wine_recognition', 'analcatdata_bankruptcy', 'soybean', 'parity5+5', 'mfeat_karhunen', 'breast_cancer_wisconsin', 'mfeat_pixel', 'house_votes_84', 'dna', 'spambase', 'iris', 'ring', 'mfeat_factors', 'splice', 'breast', 'dermatology', 'cars', 'car', 'twonorm', 'segmentation', 'monk3', 'vote', 'optdigits', 'ann_thyroid', 'analcatdata_authorship', 'pendigits', 'kr_vs_kp', 'nursery', 'collins', 'texture', 'car_evaluation', 'chess', 'agaricus_lepiota', 'analcatdata_creditscore', 'clean1', 'clean2', 'corral', 'irish', 'mofn_3_7_10', 'monk1', 'mushroom', 'mux6', 'new_thyroid', 'prnn_crabs', 'threeOf9', 'tic_tac_toe', 'xd6']

samples, features, classes = [], [], []

for ds in datasets:
    X, y = fetch_data(ds, return_X_y=True, local_cache_dir='../datasets/pmlb')
    samples.append(X.shape[0])
    features.append(X.shape[1])
    classes.append(len(np.unique(y)))

print('samples', min(samples), max(samples))
print('features', min(features), max(features))
print('classes', min(classes), max(classes))

n_bins = 6
# for cmap_category, cmap_list in cmaps:
#     for name in cmap_list:
# name = 'Accent'
# cm = plt.get_cmap(name)
# colors = [cm(i/n_bins) for i in range(5)]

for vals in [samples, features, classes]:
    ax = plt.subplot()
    title = 'samples' if vals == samples else 'features' if vals == features else 'classes'
    ax.set_title(title)
    ax.set_xlabel(f'number of {title}')
    ax.hist(vals, n_bins, density=False, histtype='bar', align='mid')  # , color=colors)  # , label=['All', '2 stages', '3 stages','4 stages', '5 stages'])
    # ax.legend(prop={'size': 10})
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    ax.set_yticks([])
    plt.show()
'''