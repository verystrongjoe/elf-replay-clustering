from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


def train_model(model, train_dataset_iter, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=argslist.lr)
  criterion = nn.MSELoss(reduction='mean').to(device)
  model.train()
  train_losses = []
  for epoch in range(1, n_epochs + 1):
    epoch_loss = 0.
    for it, seq_true in enumerate(train_dataset_iter):
      seq_true = seq_true["input"]
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred, _ = model(seq_true, seq_true.shape[0])
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    print(f"{epoch} epoch loss : {epoch_loss / 12002}")
    train_losses.append(epoch_loss)

  with open('model.pt', 'wb') as f:
    torch.save(model, f)
  return train_losses


def get_ouput_values_from_encoder(model, dataset_iter):
  model = model.eval()
  latents = []
  types = []
  with torch.no_grad():
    for it, seq_true in enumerate(dataset_iter):
      btypes = seq_true["type"]
      inputs = seq_true["input"]
      inputs = inputs.to(device)
      _, encoder_output = model(inputs, inputs.shape[0])
      latents.append(encoder_output[:,-1].detach().cpu().numpy())
      types.extend(btypes)
  latents = np.concatenate(latents, axis=0)
  types = np.asarray(types)
  return latents, types


def make_batch(samples):
  inputs = [torch.tensor(sample[0]) for sample in samples]
  types = [sample[1].item() for sample in samples]
  padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
  return {'input': padded_inputs.contiguous(), 'type':types}


dataset = CustomDataset()
X_train_iter = DataLoader(dataset, batch_size=64, collate_fn=make_batch)

model = RecurrentAutoencoder(argslist.n_features, 20, 10)
model = model.to(device)

loss = train_model(
  model,
  X_train_iter,
  n_epochs=argslist.n_epoch
)

with open('model.pt', 'rb') as f:
  model = torch.load(f)

model.eval()

i, l = get_ouput_values_from_encoder(model, X_train_iter)

np.save('input.dat', i)
np.save('label.dat', l)