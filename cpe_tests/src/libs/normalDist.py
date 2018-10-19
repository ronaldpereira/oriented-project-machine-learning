import numpy as np
import matplotlib.pyplot as plt

def create_normal_dist(mu=0, sigma=0.1, size=1000):
    dist = np.random.normal(mu, sigma, size)

    return dist

def plot_normal_dist(dist):

    mu = np.mean(dist)
    sigma = np.std(dist, ddof=1)

    _, bins, _ = plt.hist(dist, 100, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r', label='Normal distribution fit')
    plt.title('Random normal distribution with mu=%.2f and sigma=%.2f'%(mu, sigma))
    plt.legend(loc='best')
    plt.show()
