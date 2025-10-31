import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import HPCNet
from data import SplitCIFAR100, SplitImageNetR, CORe50, MedStream7kContinual
from utils import FixedSizeBuffer, spa_loss, set_seed, accuracy
from tqdm import tqdm

def get_dataset(name, **kwargs):
    if name == "split_cifar100":
        return SplitCIFAR100(**kwargs)
    elif name == "split_imagenet_r":
        return SplitImageNetR(**kwargs)
    elif name == "core50":
        return CORe50(**kwargs)
    elif name == "medstream7k":
        return MedStream7kContinual(num_tasks=kwargs.get("num_tasks", 7))
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["log_dir"])

    dataset = get_dataset(cfg["dataset"], data_dir="./data", num_tasks=cfg["num_tasks"], seed=cfg["seed"])
    buffer = FixedSizeBuffer(cfg["buffer_size"])

    initial_classes = dataset.class_names_per_task[0]
    model = HPCNet(initial_classes, embed_dim=cfg["embed_dim"], K_F=cfg["K_F"]).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs_per_task"] * cfg["num_tasks"])

    all_seen_classes = []
    task_accuracies = []

    for task_id in range(dataset.num_tasks):
        print(f"Starting Task {task_id}")
        task_data = dataset.get_task(task_id)
        loader = torch.utils.data.DataLoader(task_data, batch_size=cfg["batch_size"], shuffle=True)

        for epoch in range(cfg["epochs_per_task"]):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch}"):
                if isinstance(batch, dict):
                    images = [img.convert("RGB") for img in batch["image"]]
                    labels = batch["object_id"]
                else:
                    images, labels = batch
                    labels = [l.item() for l in labels]

                class_ids = [f"class_{l}" for l in labels]
                for cid in class_ids:
                    if cid not in all_seen_classes:
                        all_seen_classes.append(cid)

                for img, cid in zip(images, class_ids):
                    buffer.add(cid, img)

                logits, cp_bank, ip_bank, top_idxs = model(images, all_seen_classes)
                targets = torch.tensor([all_seen_classes.index(cid) for cid in class_ids]).to(device)
                loss_ce = F.cross_entropy(logits, targets)
                loss_consist = model.consistency_loss(top_idxs, cp_bank, ip_bank)
                loss = loss_ce + 0.01 * loss_consist

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            writer.add_scalar(f"Loss/Task_{task_id}", epoch_loss / len(loader), epoch)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, dict):
                    images, labels = batch
                    labels = [l.item() for l in labels]
                class_ids = [f"class_{l}" for l in labels]
                logits, _, _, _ = model(images, all_seen_classes)
                targets = torch.tensor([all_seen_classes.index(cid) for cid in class_ids]).to(device)
                acc = accuracy(logits, targets)
                correct += acc * len(targets)
                total += len(targets)
        task_acc = correct / total
        task_accuracies.append(task_acc)
        writer.add_scalar("Accuracy/Task", task_acc, task_id)

        if task_id + 1 < dataset.num_tasks:
            new_classes = dataset.class_names_per_task[task_id + 1]
            for cls in new_classes:
                model.add_new_class(cls)

        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_seen_classes': all_seen_classes,
            'task_accuracies': task_accuracies
        }, os.path.join(cfg["checkpoint_dir"], f"checkpoint_task_{task_id}.pth"))

    writer.close()
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
