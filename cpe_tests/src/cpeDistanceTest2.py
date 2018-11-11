import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulib.pu_data import pnu_from_dataframe
from pywsl.cpe.cpe_ene import cpe
from scipy.spatial.distance import euclidean

import libs.normalDist as normalDist

n_clusters = 2

clusters_config = [{} for _ in range(n_clusters)]
clustersx = [[] for _ in range(n_clusters)]
clustersy = [[] for _ in range(n_clusters)]

for index in range(n_clusters):
    if index == 0:
        clusters_config[index]['xmu'] = 0.0
        clusters_config[index]['ymu'] = 0.0
    elif index == 1:
        clusters_config[index]['xmu'] = 10.0
        clusters_config[index]['ymu'] = 0.0

    # Increasing the sigmas will cause a completely different behaviour in CPE
    clusters_config[index]['xsigma'] = 0.1
    clusters_config[index]['ysigma'] = 0.1
    clusters_config[index]['size'] = 1000

errorsList = []
eucDists = []

xmus = [x/4 for x in reversed(range(41))]

for actualxmu in xmus:
    clusters_config[index]['xmu'] = actualxmu

    eucDist = euclidean([clusters_config[0]['xmu'], clusters_config[0]['ymu']], [clusters_config[1]['xmu'], clusters_config[1]['ymu']])

    for index in range(n_clusters):
        clustersx[index] = normalDist.create_normal_dist(clusters_config[index]['xmu'], clusters_config[index]['xsigma'], clusters_config[index]['size'])
        clustersy[index] = normalDist.create_normal_dist(clusters_config[index]['ymu'], clusters_config[index]['ysigma'], clusters_config[index]['size'])

        plt.scatter(clustersx[index], clustersy[index], label='cluster %d' %index)

    plt.legend(loc='best')
    plt.savefig('../output/'+str(n_clusters)+'clusters_dist' + str(round(eucDist, 2)) + '.png')
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
    plt.savefig('../output/' + str(n_clusters) + 'clusters_all_clusters_dist' + str(round(eucDist, 2)) + '.png')
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

    errorsList.append(abs(0.5 - np.mean(priors)))
    eucDists.append(round(eucDist, 2))

print()
print(eucDists)
print(errorsList)

plt.plot(eucDists, errorsList, 'b-', label='error = real-prior')
plt.title('Errors for each euclidean distances between 2 clusters')
plt.xlabel('Euclidean Distance')
plt.ylabel('Error')
plt.legend(loc='best')
plt.savefig('../output/' + str(n_clusters) + 'clusters_errors_euc_distances.png')
plt.clf()
