import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulib.pu_data import pnu_from_dataframe
from pywsl.cpe.cpe_ene import cpe
from scipy.spatial.distance import euclidean

import libs.normalDist as normalDist

totalRepetitions = 30

repErrors = []
for repetition in range(totalRepetitions):
    print('Executing repetition %d of %d total' %(repetition+1, totalRepetitions))

    n_clusters = 3

    clusters_config = [{} for _ in range(n_clusters)]
    clustersx = [[] for _ in range(n_clusters)]
    clustersy = [[] for _ in range(n_clusters)]

    for index in range(n_clusters):
        if index == 0:
            clusters_config[index]['xmu'] = 0.0
            clusters_config[index]['ymu'] = 0.0
        elif index == 1:
            clusters_config[index]['xmu'] = 5.0
            clusters_config[index]['ymu'] = 0.0
        elif index == 2:
            clusters_config[index]['xmu'] = 0.0
            clusters_config[index]['ymu'] = 5.0

        # Increasing the sigmas will cause a completely different behaviour in CPE
        clusters_config[index]['xsigma'] = 1.0
        clusters_config[index]['ysigma'] = 1.0
        clusters_config[index]['size'] = 1000

    errorsList = []
    eucDists = []

    xymus = [x/4 for x in reversed(range(21))]

    for actualxymu in xymus:
        clusters_config[1]['xmu'] = actualxymu
        clusters_config[2]['ymu'] = actualxymu

        eucDist = euclidean([clusters_config[0]['xmu'], clusters_config[0]['ymu']], [clusters_config[1]['xmu'], clusters_config[1]['ymu']])

        for index in range(n_clusters):
            clustersx[index] = normalDist.create_normal_dist(clusters_config[index]['xmu'], clusters_config[index]['xsigma'], clusters_config[index]['size'])
            clustersy[index] = normalDist.create_normal_dist(clusters_config[index]['ymu'], clusters_config[index]['ysigma'], clusters_config[index]['size'])

            plt.scatter(clustersx[index], clustersy[index], label='cluster %d' %index)

        plt.legend(loc='best')
        # plt.savefig('../output/'+str(n_clusters)+'clusters_dist' + str(round(eucDist, 2)) + '.png')
        plt.clf()

        xclusters = []
        yclusters = []
        labelclusters = []
        for index in range(n_clusters):
            xclusters.extend(clustersx[index])
            yclusters.extend(clustersy[index])
            labelclusters.extend([index for _ in range(clusters_config[index]['size'])])

        plt.hexbin(xclusters, yclusters, gridsize=25, cmap='inferno')
        plt.title('All clusters density plot')
        cb = plt.colorbar()
        cb.set_label('counts')
        # plt.savefig('../output/' + str(n_clusters) + 'clusters_all_clusters_dist' + str(round(eucDist, 2)) + '.png')
        plt.clf()

        print()
        
        priors = []
        for index in range(n_clusters):
            df = pd.DataFrame({'x_value':xclusters, 'y_value':yclusters, 'label':labelclusters})

            df = pnu_from_dataframe(df, 'label', index, pos_size=0.8, neg_size=0.8)

            x_l = df.loc[df['y'] != 0][['x_value', 'y_value']].values
            y_l = df.loc[df['y'] != 0]['y'].values
            x_u = df.loc[df['y'] == 0][['x_value', 'y_value']].values

            prior = cpe(x_l, y_l, x_u)

            priors.append(prior)

            print('cpe prior for %d with euclidean distance %.3f = %.8f' %(index, eucDist, prior))

        errorsList.append(np.mean([abs(1/3 - x) for x in priors]))
        eucDists.append(round(eucDist, 2))

    print()

    for dist, error in zip(eucDists, errorsList):
        print('error for euclidean distance %.3f : %f' %(dist, error), end='\n\n')

    plt.plot(eucDists, errorsList, 'b-', label='error = real-prior')
    plt.title('Errors for each euclidean distances between 2 clusters')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Error')
    plt.legend(loc='best')
    # plt.savefig('../output/' + str(n_clusters) + 'clusters_errors_euc_distances.png')
    plt.clf()

    repErrors.append(errorsList)

repErrorsMean = np.mean(repErrors, axis=0)
repErrorsStd = np.std(repErrors, axis=0)

plt.figure(figsize=(18, 6))

plt.plot(eucDists, repErrorsMean, 'b-o', label='CPE mean error')

(_, caps, _) = plt.errorbar(eucDists, repErrorsMean, yerr=repErrorsStd, fmt='bo', capsize=5 , label='CPE Standard Deviation')
for cap in caps:
    cap.set_markeredgewidth(1)

plt.title('CPE errors for %d clusters with %d repetitions' %(n_clusters, totalRepetitions))
plt.xlabel('Euclidean Distance')
plt.ylabel('Mean Error')
plt.legend(loc='best')
plt.savefig('../output/cpeDistTest/%dClusters%dRepetitions.pdf' %(n_clusters, totalRepetitions))
plt.clf()
