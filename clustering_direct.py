from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
from sklearn.cluster import KMeans
import numpy as np
import os

with open('model.pt', 'rb') as f:
    model = torch.load(f)

model.eval()


for i, f in enumerate(datalist):
    if i > 5:
        continue
    fn = os.path.join(root_dir, datalist[i])
    sample = np.load(fn)
    x1 = torch.tensor(sample['state']).sum(axis=1).flatten(start_dim=1)
    Xs = torch.cat([Xs, x1], axis=1)

    splits = datalist[i].split('_')
    if splits[1] =='hit_run':
        if splits[2] == '20':
            type = 1
        elif splits[2] == '50':
            type = 2
        elif splits[3] == '80':
            type = 3
    elif splits[1] =='simple':
        if splits[2] == '20':
            type = 4
        elif splits[2] == '50':
            type = 5
        elif splits[3] == '80':
            type = 6
    labels.append(type)

_, latents = model(Xs, 12002)
latents = latents.deatch().item()


# 모델 예측치
kmeans = KMeans(n_clusters=6, random_state=0).fit(latents)
kmeans.labels_

cnt = np.zeros(6,6)

# 각 예측된 분류에서 클러스터링 비율 구하기
for i, j in zip(latents, kmeans.labels_):
    cnt[i][j] = cnt[i][j] + 1

# 컨퓨전메트릭스 만들기
print(cnt)