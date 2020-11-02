from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
from sklearn.cluster import KMeans
import numpy as np
import os

latents = np.load('input.dat.npy')
labels = np.load('label.dat.npy')


# 모델 예측치
kmeans = KMeans(n_clusters=6, random_state=0).fit(latents)
kmeans.labels_

cnt = np.zeros((6, 6))

# 각 예측된 분류에서 클러스터링 비율 구하기
for i, j in zip(labels, kmeans.labels_):
    cnt[i-1][j-1] = cnt[i-1][j-1] + 1

# 컨퓨전메트릭스 만들기
print(cnt)