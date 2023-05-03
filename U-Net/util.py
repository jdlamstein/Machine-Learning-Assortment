import torch
from torchvision.utils import save_image
import os


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Found {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()
    return num_correct / num_pixels * 100, dice_score / len(loader)


def save_predictions_as_imgs(loader, model, save_folder, device="cuda",
                             save_masks=True):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        im = x.detach().clone()
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        save_image(
            preds, os.path.join(save_folder, f'prediction_{idx}.png')
        )
        if save_masks:
            save_image(y.unsqueeze(1), os.path.join(save_folder, f'gt_{idx}.png'))
        else:
            save_image(im, os.path.join(save_folder, f'crop_{idx}.png'))
    model.train()
