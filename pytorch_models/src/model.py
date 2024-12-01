"""Sample model code for PyTorch flavor. The model if is working with float32, needs to convert explicitly to float32."""

import torch
from tqdm import tqdm

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: converting to float32 since mlflow will give us float64 at inference time
        x = x.float()
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, feat_size: int, target_size: int, num_samples: int = 1000):
        self.data = torch.randn(num_samples, feat_size)
        self.target = torch.randn(num_samples, target_size)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
class Trainer():
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataset: torch.utils.data.Dataset, epochs: int = 5) -> torch.nn.Module:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        for epoch in range(epochs):
            batch_loss = 0
            for x, y in (pbar := tqdm(dataloader)):
                batch_loss += self.train_step(x, y)/len(dataloader)
                pbar.set_description(f"loss: {batch_loss:.2f}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {batch_loss:.2f}")
        return self.model