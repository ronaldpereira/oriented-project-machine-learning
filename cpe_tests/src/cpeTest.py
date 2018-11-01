import matplotlib.pyplot as plt
import pandas as pd

import libs.normalDist as normalDist
from pywsl.cpe.cpe_ene import cpe
from pulib.pu_data import pnu_from_dataframe

n_clusters = int(input('Enter the number of clusters to create: '))

clusters_config = [{} for _ in range(n_clusters)]
clustersx = [[] for _ in range(n_clusters)]
clustersy = [[] for _ in range(n_clusters)]

for index in range(n_clusters):
    print('Enter cluster %d config:' %index)
    clusters_config[index]['xmu'] = float(input('mu axis x = '))
    clusters_config[index]['ymu'] = float(input('mu axis y = '))
    clusters_config[index]['xsigma'] = float(input('sigma axis x = '))
    clusters_config[index]['ysigma'] = float(input('sigma axis y = '))
    clusters_config[index]['size'] = int(input('size = '))

print('\n')
print(clusters_config)

for index in range(n_clusters):
    clustersx[index] = normalDist.create_normal_dist(clusters_config[index]['xmu'], clusters_config[index]['xsigma'], clusters_config[index]['size'])
    clustersy[index] = normalDist.create_normal_dist(clusters_config[index]['ymu'], clusters_config[index]['ysigma'], clusters_config[index]['size'])

    plt.scatter(clustersx[index], clustersy[index], label='cluster %d' %index)

plt.legend(loc='best')
plt.savefig('../output/'+str(n_clusters)+'clusters.png')
plt.clf()

for index in range(n_clusters):
    plt.hexbin(clustersx[index], clustersy[index], gridsize=25, cmap='inferno')
    plt.title('Cluster ' + str(index) + ' density plot')
    cb = plt.colorbar()
    cb.set_label('counts')
    plt.savefig('../output/'+str(n_clusters)+'clusters_cluster'+str(index)+'.png')
    plt.clf()

xclusters = []
yclusters = []
labelclusters = []
for index in range(n_clusters):
    xclusters.extend(clustersx[index])
    yclusters.extend(clustersy[index])
    labelclusters.extend([index for _ in range(clusters_config[index]['size'])])

df = pd.DataFrame({'x_value':xclusters, 'y_value':yclusters, 'label':labelclusters})

df = pnu_from_dataframe(df, 'label', 0, pos_size=0.8, neg_size=0)

x_l = df.loc[df['y'] != 0][['x_value', 'y_value']].values
y_l = df.loc[df['y'] != 0]['y'].values
x_u = df.loc[df['y'] == 0][['x_value', 'y_value']].values

prior = cpe(x_l, y_l, x_u)

print('cpe prior = ', prior)
