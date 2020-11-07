import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# home_dir = '/workspace/elf-replay-clustering'
home_dir = 'D:/workspace/elf-replay-clustering'
batch_size = 128
lr = 1e-2
dropout = 0.1
n_epoch = 1000
n_features = 400

num_negatives = 8192
num_raw_channels = 22

#categorical_feature_idxes = [0,1,2,3,4,5,17,18,19]
#numerical_feature_idxes = [i for i in range(1, 22) if i not in categorical_feature_idxes]
numerical_feature_idxes = [6, 7]
categorical_feature_idxes = [0, 1, 2, 3, 4, 5, 17, 18, 19]
feature_idxes = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19]


# sampling_length = 10  #  리플레이별 positive pair 각 길이
sampling_tuple_idx_1 = 1   # positive pair 처음꺼
sampling_tuple_idx_2 = 30  # positive pair 두번째꺼

sample_len_threshold = 50  # 위의 샘플을 구하기 위해서 최소한 40 time seq이상의 데이터만 취함
embed_dim = 5 # 모든 채널이 embedding을 개별로 가지나 출력 dim은 통일
proj_dim = 128
window_size = 20

temperature = 0.1
