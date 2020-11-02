from torch.utils.data import DataLoader
from my_dataset import CustomDataset
import torch

def make_batch(samples):
    inputs = [torch.tensor(sample) for sample in samples]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return {'input': padded_inputs.contiguous()}

dataset = CustomDataset('model_45252', "simple", 50)

data_loader = DataLoader(dataset, batch_size=5, collate_fn=make_batch)
data_loader_iterator = iter(data_loader)

for idx in range(5):
    sample = data_loader_iterator.next()
    print(f"{idx} shape =  {sample['input'].shape}")
