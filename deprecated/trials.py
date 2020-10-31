import torch
input = torch.randn(64, 22, 98)
encoder = torch.nn.LSTM(98, 256, batch_first=True)
encoded, _ = encoder(input)
decoder_input = encoded[:, -1:].repeat(1, 10, 1)
decoder = torch.nn.LSTM(256, 98, batch_first=True)
decoded, _ = decoder(decoder_input)
print(decoded.shape) #torch.Size([64, 10, 98])