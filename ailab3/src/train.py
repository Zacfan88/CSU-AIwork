import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleCNN


def train_and_eval(train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   num_classes: int,
                   epochs: int = 10,
                   lr: float = 1e-3,
                   optimizer_name: str = 'adam',
                   activation: str = 'relu',
                   device: str = None) -> Tuple[float, str]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=num_classes, activation=activation).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    log_lines = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        # Eval
        model.eval()
        with torch.no_grad():
            total_t = 0
            correct_t = 0
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_t += targets.size(0)
                correct_t += (predicted == targets).sum().item()
            test_acc = correct_t / total_t if total_t else 0.0

        best_acc = max(best_acc, test_acc)
        line = f"Epoch {epoch:03d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
        print(line)
        log_lines.append(line)

    # Save model & logs
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('outputs')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'model_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)

    log_path = os.path.join(out_dir, f'log_{timestamp}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    return best_acc, model_path