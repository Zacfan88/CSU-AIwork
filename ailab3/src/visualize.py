# -*- coding: utf-8 -*-
import argparse
import os
import re
import time
from typing import Tuple, List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # ??????/?????????
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ??????????????????????????????????
# ?? macOS ????? 'PingFang SC' ?? 'Hiragino Sans GB'??????????????
try:
    plt.rcParams['font.sans-serif'] = [
        'PingFang SC',
        'Hiragino Sans GB',
        'Noto Sans CJK SC',
        'SimHei',
        'Microsoft YaHei',
        'Arial Unicode MS',
        'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    # ????????????????????????
    pass

from data import (
    get_digits_letters_datasets,
    get_chinese_datasets,
    _build_transform,
    _list_classes,
    _is_letter_class,
    _is_chinese_char,
    FilteredImageFolder,
)
from models import SimpleCNN


def parse_log_file(log_path: str) -> dict:
    epochs: List[int] = []
    losses: List[float] = []
    train_accs: List[float] = []
    test_accs: List[float] = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m_epoch = re.search(r'Epoch\s+(\d+)', line)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group(1))
            m_loss = re.search(r'loss=([0-9.]+)', line)
            m_train = re.search(r'(?:train_acc|in_acc)=([0-9.]+)', line)
            m_test = re.search(r'test_acc=([0-9.]+)', line)
            if m_loss and m_train and m_test:
                epochs.append(epoch)
                losses.append(float(m_loss.group(1)))
                train_accs.append(float(m_train.group(1)))
                test_accs.append(float(m_test.group(1)))
    return {
        'epochs': epochs,
        'loss': losses,
        'train_acc': train_accs,
        'test_acc': test_accs,
    }


def plot_training_curves(metrics: dict, out_path: str, max_epoch: int | None = None):
    if not metrics['epochs']:
        print('No metrics parsed from log; skip training curves.')
        return
    epochs = metrics['epochs']
    if max_epoch is not None:
        idxs = [i for i, e in enumerate(epochs) if e <= max_epoch]
        if not idxs:
            print(f'No epochs <= {max_epoch}; skip curves.')
            return
        def take(lst):
            return [lst[i] for i in idxs]
        epochs = take(metrics['epochs'])
        metrics = {
            'epochs': epochs,
            'loss': take(metrics['loss']),
            'train_acc': take(metrics['train_acc']),
            'test_acc': take(metrics['test_acc']),
        }
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['loss'], label='loss', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'], label='train_acc', color='tab:blue')
    plt.plot(epochs, metrics['test_acc'], label='test_acc', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Training curves saved: {out_path}')


def build_mapping_digits_letters(base_dir: str) -> List[str]:
    num_train = os.path.join(base_dir, 'dataset_num', 'train_28_28')
    num_test = os.path.join(base_dir, 'dataset_num', 'testset_28_28')
    digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
    # ?????????? dataset_num??train ?? test ????????????????
    union_letter_dirs = sorted(set(_list_classes(num_train)) | set(_list_classes(num_test)))
    letter_classes = [c for c in union_letter_dirs if _is_letter_class(c)]
    letters_sorted = sorted(letter_classes)
    digits_sorted = sorted(digits_classes)
    return letters_sorted + digits_sorted


def build_mapping_chinese(base_dir: str) -> List[str]:
    char_train = os.path.join(base_dir, 'dataset_char', 'train_28_28')
    char_test = os.path.join(base_dir, 'dataset_char', 'test_28_28')
    union = sorted(set(_list_classes(char_train)) | set(_list_classes(char_test)))
    chinese_classes = [c for c in union if _is_chinese_char(c)]
    return sorted(chinese_classes)


def compute_confusion_matrix(task: str,
                             data_root: str,
                             model_path: str,
                             img_size: int,
                             activation: str,
                             batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if task == 'digits_letters':
        _, test_loader, num_classes = get_digits_letters_datasets(data_root, img_size=img_size, batch_size=batch_size)
        labels = build_mapping_digits_letters(data_root)
    else:
        _, test_loader, num_classes = get_chinese_datasets(data_root, img_size=img_size, batch_size=batch_size)
        labels = build_mapping_chinese(data_root)

    model = SimpleCNN(num_classes=num_classes, activation=activation).to(device)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # ?????? num_classes ?????
        num_classes_saved = None
        if 'classifier.4.weight' in state:
            num_classes_saved = state['classifier.4.weight'].shape[0]
        else:
            for k, v in state.items():
                if k.endswith('weight') and getattr(v, 'dim', lambda: 0)() == 2:
                    num_classes_saved = v.shape[0]
                    break
        if num_classes_saved and num_classes_saved != num_classes:
            model = SimpleCNN(num_classes=num_classes_saved, activation=activation).to(device)
            model.load_state_dict(state)
            # ?? digits_letters ??????????10????????????????????
            if task == 'digits_letters' and num_classes_saved == 10:
                labels = [str(i) for i in range(10)]
                # ??????????????????????????????????????
                num_test = os.path.join(data_root, 'dataset_num', 'testset_28_28')
                digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_test, str(i)))]
                tfm_test = _build_transform(img_size)
                test_digits = FilteredImageFolder(num_test, digits_classes, class_to_idx={c: i for i, c in enumerate(sorted(digits_classes))}, transform=tfm_test)
                test_loader = torch.utils.data.DataLoader(test_digits, batch_size=batch_size, shuffle=False, num_workers=0)
            else:
                labels = labels[:num_classes_saved]
        else:
            raise

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                ti = int(t.item())
                pi = int(p.item())
                if 0 <= ti < cm.shape[0] and 0 <= pi < cm.shape[1]:
                    cm[ti, pi] += 1

    return cm, labels


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str):
    n = len(labels)
    # ??????????????????
    w = min(12, max(6, n * 0.28))
    h = min(10, max(6, n * 0.28))
    plt.figure(figsize=(w, h))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # ?????????????????
    if n <= 40:
        plt.xticks(ticks=np.arange(n) + 0.5, labels=labels, rotation=90)
        plt.yticks(ticks=np.arange(n) + 0.5, labels=labels, rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Confusion matrix saved: {out_path}')


def save_augmented_samples(task: str, data_root: str, img_size: int, out_path: str, n_samples: int = 24):
    # ?????????????????
    if task == 'digits_letters':
        train_loader, _, _ = get_digits_letters_datasets(data_root, img_size=img_size, batch_size=32, augment=True)
    else:
        train_loader, _, _ = get_chinese_datasets(data_root, img_size=img_size, batch_size=32, augment=True)

    imgs = []
    for batch_imgs, _ in train_loader:
        for i in range(batch_imgs.size(0)):
            if len(imgs) >= n_samples:
                break
            x = batch_imgs[i].squeeze(0).detach().cpu().numpy()  # [H,W]
            # ?????????mean=0.5, std=0.5
            x = x * 0.5 + 0.5
            imgs.append(x)
        if len(imgs) >= n_samples:
            break

    if not imgs:
        print('No images collected for augmentation preview; skip.')
        return

    # ????????
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    plt.figure(figsize=(cols * 2.0, rows * 2.0))
    for idx, img in enumerate(imgs):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle('Augmented Training Samples', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Augmentation samples saved: {out_path}')


def find_latest_by_prefix(out_dir: str, prefix: str, ext: str | None = None) -> str | None:
    best = None
    best_ts = ''
    for fname in os.listdir(out_dir):
        if not fname.startswith(prefix):
            continue
        if ext and not fname.endswith(ext):
            continue
        parts = fname.split('_')
        if len(parts) >= 3:
            ts = parts[-1].split('.')[0]
            if ts > best_ts:
                best_ts = ts
                best = os.path.join(out_dir, fname)
    return best


def main():
    parser = argparse.ArgumentParser(description='Visualize training logs, confusion matrix, and augmentation samples')
    parser.add_argument('--task', choices=['digits_letters', 'chinese'], required=True, help='Task to visualize')
    parser.add_argument('--data_root', default=os.path.join(os.getcwd(), 'dataset'), help='Dataset root directory')
    parser.add_argument('--log_path', default=None, help='Path to log file (Epoch .. lines)')
    parser.add_argument('--model_path', default=None, help='Path to saved model (.pt)')
    parser.add_argument('--img_size', type=int, default=28, help='Input image size')
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu'], default='elu', help='Activation for model construction')
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'visuals'), help='Directory to save plots')
    parser.add_argument('--max_epoch', type=int, default=None, help='Plot curves up to this epoch')
    parser.add_argument('--no_curves', action='store_true', help='Skip training curves')
    parser.add_argument('--no_cm', action='store_true', help='Skip confusion matrix')
    parser.add_argument('--no_aug', action='store_true', help='Skip augmentation samples')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ??????????/?????????????
    if not args.log_path:
        latest_log = find_latest_by_prefix('outputs', 'log_', ext='.txt')
        if latest_log:
            args.log_path = latest_log
    if not args.model_path:
        # ????????????????????
        if args.log_path and os.path.basename(args.log_path).startswith('log_'):
            ts = os.path.basename(args.log_path)[4:].split('.')[0]
            cand = os.path.join('outputs', f'model_{ts}.pt')
            args.model_path = cand if os.path.exists(cand) else find_latest_by_prefix('outputs', 'model_', ext='.pt')
        else:
            args.model_path = find_latest_by_prefix('outputs', 'model_', ext='.pt')

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # ???????
    if not args.no_curves and args.log_path and os.path.exists(args.log_path):
        metrics = parse_log_file(args.log_path)
        out_curves = os.path.join(args.output_dir, f'curves_{timestamp}.png')
        plot_training_curves(metrics, out_curves, max_epoch=args.max_epoch)
    else:
        print('Skip curves: log file not provided or not found.')

    # ????????
    if not args.no_cm and args.model_path and os.path.exists(args.model_path):
        cm, labels = compute_confusion_matrix(args.task, args.data_root, args.model_path, args.img_size, args.activation)
        out_cm = os.path.join(args.output_dir, f'cm_{args.task}_{timestamp}.png')
        plot_confusion_matrix(cm, labels, out_cm)
    else:
        print('Skip confusion matrix: model file not provided or not found.')

    # ???????
    if not args.no_aug:
        out_aug = os.path.join(args.output_dir, f'aug_{args.task}_{timestamp}.png')
        save_augmented_samples(args.task, args.data_root, args.img_size, out_aug, n_samples=24)
    else:
        print('Skip augmentation samples by flag.')


if __name__ == '__main__':
    main()