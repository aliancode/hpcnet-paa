
import torch
from models import HPCNet
from data import SplitCIFAR100
from utils import accuracy, compute_owcls

def compute_A_t(model, dataset, all_classes):
    all_data = dataset.get_task(0)
    for t in range(1, dataset.num_tasks):
        all_data = torch.utils.data.ConcatDataset([all_data, dataset.get_task(t)])
    loader = torch.utils.data.DataLoader(all_data, batch_size=128, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            class_ids = [f"class_{l.item()}" for l in labels]
            logits, _, _, _ = model(images, all_classes)
            targets = torch.tensor([all_classes.index(cid) for cid in class_ids])
            acc = accuracy(logits, targets)
            correct += acc * len(targets)
            total += len(targets)
    return correct / total

def load_zero_shot_dataset(dataset_name):
    if dataset_name == "imagenet-a":
        return datasets.ImageFolder("./data/imagenet-a", transform=...)
    elif dataset_name == "medstream-zs":
        return MedStreamZS()  # Real zero-shot medical classes

def compute_A_zs(model, zero_shot_classes, zs_dataset):
    loader = torch.utils.data.DataLoader(zs_dataset, batch_size=128)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            class_ids = [f"class_{l}" for l in labels]
            logits, _, _, _ = model(images, zero_shot_classes)
            targets = torch.tensor([zero_shot_classes.index(cid) for cid in class_ids])
            acc = accuracy(logits, targets)
            correct += acc * len(targets)
            total += len(targets)
    return correct / total

def compute_BWT(task_accuracies):
    if len(task_accuracies) < 2:
        return 0.0
    bwt = 0.0
    for i in range(len(task_accuracies) - 1):
        bwt += task_accuracies[-1] - task_accuracies[i]
    return bwt / (len(task_accuracies) - 1)

def main(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = HPCNet(ckpt['all_seen_classes'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    A_t = compute_A_t(model, SplitCIFAR100(), ckpt['all_seen_classes'])
    A_zs = compute_A_zs(model, ["airplane", "car"])
    BWT = compute_BWT(ckpt['task_accuracies'])
    OWCLS = compute_owcls(A_t, A_zs, BWT)
    print(f"A_t: {A_t:.4f}, A_zs: {A_zs:.4f}, BWT: {BWT:.4f}, OWCLS: {OWCLS:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <checkpoint.pth>")
        sys.exit(1)
    main(sys.argv[1])
