import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 0, 0.1
size = 10000

s = np.random.normal(mu, sigma, size)

count, bins, ignored = plt.hist(s, 100, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r', label='Normal distribution fit')
plt.title('Random normal distribution with mu=%.2f and sigma=%.2f'%(mu, sigma))
plt.legend(loc='best')
plt.show()
