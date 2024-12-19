from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris['data']
y = iris['target']
# Створення KMeans
kmeans = KMeans( n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, algorithm='lloyd' )

# Навчання моделі кластеризації
kmeans.fit(X)
# Прогнозування кластерів
y_kmeans = kmeans.predict(X)
# Побудова графіку кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# Визначення функції пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed) # Генератор випадкових чисел
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers) # Призначення міток на основі мінімальної відстані
        new_centers = np.array([X[labels == i].mean(0)
        for i in range(n_clusters)]) # Оновлення центрів
        if np.all(centers == new_centers): # Якщо центри не змінюються, виходимо з циклу
            break
        centers = new_centers
    return centers, labels

centers, labels = find_clusters(X, 3)
plt.figure()
# Новий графік
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# Ще один пошук кластерів з іншим rseed
centers, labels = find_clusters(X, 3, rseed=0)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# Використання KMeans для кластеризації
labels = KMeans(3, random_state=0).fit_predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()