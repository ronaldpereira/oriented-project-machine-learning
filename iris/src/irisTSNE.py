from sklearn.manifold import TSNE
import pandas as pd
from ggplot import ggplot, geom_point, ggtitle, aes

irisDataset = pd.read_csv('../input/iris_dataset.csv')

df = irisDataset.drop(['Id','Species'],axis=1)

tsne_results = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(df)

tsne = None
tsne = irisDataset.copy()
tsne['X t-SNE'] = tsne_results[:,0]
tsne['Y t-SNE'] = tsne_results[:,1]

chart = ggplot( tsne, aes(x='X t-SNE', y='Y t-SNE', color='Species', label='Species') ) \
        + geom_point(size=70) \
        + ggtitle("t-SNE plot colored by specie")

chart.save('../output/tsne/tsneIris.png')
