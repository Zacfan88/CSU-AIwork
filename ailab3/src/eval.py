import argparse
import os
import torch
from PIL import Image

from data import (
    get_digits_letters_datasets,
    get_chinese_datasets,
    _build_transform,
    _list_classes,
    _is_letter_class,
    _is_chinese_char,
)
from models import SimpleCNN


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total else 0.0


def build_mapping_digits_letters(base_dir: str) -> list[str]:
    num_train = os.path.join(base_dir, 'dataset_num', 'train_28_28')
    num_test = os.path.join(base_dir, 'dataset_num', 'testset_28_28')
    digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
    union_letter_dirs = sorted(set(_list_classes(num_train)) | set(_list_classes(num_test)))
    letter_classes = [c for c in union_letter_dirs if _is_letter_class(c)]
    letters_sorted = sorted(letter_classes)
    digits_sorted = sorted(digits_classes)
    # ??????????????????????????????
    return letters_sorted + digits_sorted


def build_mapping_chinese(base_dir: str) -> list[str]:
    char_train = os.path.join(base_dir, 'dataset_char', 'train_28_28')
    char_test = os.path.join(base_dir, 'dataset_char', 'test_28_28')
    union = sorted(set(_list_classes(char_train)) | set(_list_classes(char_test)))
    chinese_classes = [c for c in union if _is_chinese_char(c)]
    return sorted(chinese_classes)


def predict_single_image(task: str, data_root: str, model_path: str, img_path: str, img_size: int, activation: str, device: str):
    # ???????????????
    if task == 'digits_letters':
        # ?????????????? dataset_num???????????????????????????????????????
        num_train = os.path.join(data_root, 'dataset_num', 'train_28_28')
        num_test = os.path.join(data_root, 'dataset_num', 'testset_28_28')
        digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
        union_letter_dirs = sorted(set(_list_classes(num_train)) | set(_list_classes(num_test)))
        letter_classes = [c for c in union_letter_dirs if _is_letter_class(c)]
        letters_sorted = sorted(letter_classes)
        digits_sorted = sorted(digits_classes)
        classes = letters_sorted + digits_sorted
    else:
        classes = build_mapping_chinese(data_root)
    num_classes = len(classes)

    model = SimpleCNN(num_classes=num_classes, activation=activation).to(device)
    state = torch.load(model_path, map_location=device)
    # ?????? num_classes ?????
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # ??????????????
        num_classes_saved = None
        if 'classifier.4.weight' in state:
            num_classes_saved = state['classifier.4.weight'].shape[0]
        else:
            for k, v in state.items():
                if k.endswith('weight') and v.dim() == 2:
                    num_classes_saved = v.shape[0]
                    break
        if num_classes_saved and num_classes_saved != num_classes:
            model = SimpleCNN(num_classes=num_classes_saved, activation=activation).to(device)
            model.load_state_dict(state)
            # digits-only??10??????????????????????????
            if task == 'digits_letters':
                if 'letters_sorted' in locals() and 'digits_sorted' in locals():
                    if num_classes_saved == len(digits_sorted):
                        classes = digits_sorted
                    else:
                        classes = (letters_sorted + digits_sorted)[:num_classes_saved]
                else:
                    classes = classes[:num_classes_saved]
        else:
            raise

    tfm = _build_transform(img_size)
    with Image.open(img_path) as im:
        im = im.convert('RGB')
    x = tfm(im).unsqueeze(0).to(device)  # [1,1,H,W]

    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred_idx = int(prob.argmax(dim=1).item())
        conf = float(prob[0, pred_idx].item())
    pred_cls = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
    print(f"Predict: {pred_cls} (index={pred_idx}, confidence={conf:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved CNN model on test set or single image')
    parser.add_argument('--task', choices=['digits_letters', 'chinese'], required=True, help='Which dataset/task to evaluate')
    parser.add_argument('--data_root', default=os.path.join(os.getcwd(), 'dataset'), help='Dataset root directory')
    parser.add_argument('--model_path', required=True, help='Path to saved model .pt file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--img_size', type=int, default=28, help='Input image size')
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu', help='Activation function (should match training)')
    parser.add_argument('--predict_image', type=str, default=None, help='Predict a single image instead of evaluating the whole test set')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.predict_image:
        predict_single_image(args.task, args.data_root, args.model_path, args.predict_image, args.img_size, args.activation, device)
        return

    if args.task == 'digits_letters':
        _, test_loader, num_classes = get_digits_letters_datasets(args.data_root, img_size=args.img_size, batch_size=args.batch_size)
    else:
        _, test_loader, num_classes = get_chinese_datasets(args.data_root, img_size=args.img_size, batch_size=args.batch_size)

    model = SimpleCNN(num_classes=num_classes, activation=args.activation).to(device)

    state = torch.load(args.model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        num_classes_saved = None
        if 'classifier.4.weight' in state:
            num_classes_saved = state['classifier.4.weight'].shape[0]
        else:
            for k, v in state.items():
                if k.startswith('classifier') and k.endswith('weight') and v.dim() == 2:
                    num_classes_saved = v.shape[0]
                    break
        if num_classes_saved and num_classes_saved != num_classes:
            model = SimpleCNN(num_classes=num_classes_saved, activation=args.activation).to(device)
            model.load_state_dict(state)
        else:
            raise e

    acc = evaluate(model, test_loader, device)
    print(f'Test accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()