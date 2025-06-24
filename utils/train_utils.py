import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(-1)  # 这里改动
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float().unsqueeze(-1)  # 这里改动
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def save_results(filepath, best_params, test_loss):
    with open(filepath, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")

