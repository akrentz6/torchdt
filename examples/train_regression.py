"""Fit y = 2x with one encoded parameter and no external dataset."""
import torch
import torch.nn.functional as F
from torchdt.lns import LNS16
from torchdt.optim import SGD
from torchdt.optim.lr_scheduler import StepLR


def main():
    LNS16.set_prec(7)
    x = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]], dtype=LNS16)
    target = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=LNS16)
    model = torch.nn.Linear(1, 1, bias=False, dtype=LNS16)
    torch.nn.init.constant_(model.weight, 0.25)
    assert isinstance(model.weight, LNS16)
    optimizer = SGD(LNS16, model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    initial_loss = F.mse_loss(model(x), target).item()

    for step in range(40):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 10 == 0:
            print(f"Step {step:2d}: loss={loss.item():.5f}")

    with torch.no_grad():
        final_loss = F.mse_loss(model(x), target).item()
    print(f"Weight: {model.weight.item():.4f}; final loss: {final_loss:.5f}")
    assert final_loss < initial_loss * 0.1, "Expected a substantial loss decrease"


if __name__ == "__main__":
    main()
