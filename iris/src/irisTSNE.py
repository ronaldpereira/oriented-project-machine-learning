from sklearn.manifold import TSNE
import pandas as pd
from ggplot import *

irisDataset = pd.read_csv('../input/iris_dataset.csv')

df = irisDataset.drop(['Id','Species'],axis=1)

tsne_results = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(df)

tsne = None
tsne = irisDataset.copy()
tsne['x-tsne'] = tsne_results[:,0]
tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( tsne, aes(x='x-tsne', y='y-tsne', color='Species', label='Species') ) \
        + geom_point(size=70) \
        + ggtitle("tSNE dimensions colored by Specie")

chart.save('../output/tsne/tsneIris.png')
