import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from data import _build_transform, _list_classes, _is_letter_class, _is_chinese_char
from models import SimpleCNN


def build_mapping_digits_letters(base_dir: str) -> List[str]:
    num_train = os.path.join(base_dir, 'dataset_num', 'train_28_28')
    num_test = os.path.join(base_dir, 'dataset_num', 'testset_28_28')
    digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
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


def load_model(model_path: str, num_classes: int, activation: str, device: str) -> torch.nn.Module:
    model = SimpleCNN(num_classes=num_classes, activation=activation).to(device)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
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
        else:
            raise
    model.eval()
    return model


def order_points(pts: np.ndarray) -> np.ndarray:
    # Order 4 points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_plate(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return cropped plate region and bbox (x,y,w,h). Fallback to whole image if not found."""
    h, w = img_bgr.shape[:2]
    scale = 1280.0 / max(w, h) if max(w, h) > 1280 else 1.0
    if scale != 1.0:
        img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        img = img_bgr.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color masks for typical CN plates: blue, green, yellow
    mask_blue = cv2.inRange(hsv, (100, 70, 50), (140, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 50, 40), (85, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    mask = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_green, mask_yellow))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = None
    best_rect = None
    img_vis = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (img.shape[0] * img.shape[1]) * 0.002:  # ignore tiny areas
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        w_box, h_box = max(rw, rh), min(rw, rh)
        aspect = w_box / max(h_box, 1)
        if 2.5 <= aspect <= 6.0 and h_box > 20:
            candidate = cnt
            best_rect = rect
            break

    if candidate is None:
        # Fallback: edge-based
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < (img.shape[0] * img.shape[1]) * 0.005:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            aspect = ww / max(hh, 1)
            if 2.5 <= aspect <= 6.0 and hh > 20:
                best_rect = ((x + ww / 2, y + hh / 2), (ww, hh), 0)
                candidate = cnt
                break

    if best_rect is None:
        # give up: return entire image
        return img_bgr, (0, 0, w, h)

    box = cv2.boxPoints(best_rect)
    box = box.astype(np.int32)
    # perspective warp to canonical ratio 440x140 (approx 3.14)
    rect = order_points(box.astype('float32'))
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))

    # upscale bbox back to original scale approx
    x_min, y_min = warped.shape[1], warped.shape[0]
    return warped, (0, 0, warped.shape[1], warped.shape[0])


def segment_characters(plate_bgr: np.ndarray) -> List[np.ndarray]:
    img = plate_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # enhance characters using morphological blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Some plates may respond better without blackhat; combine
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.bitwise_or(th, th2)

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = th.shape
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if hh < h * 0.35 or hh > h * 0.95:
            continue
        if ww < w * 0.02 or ww > w * 0.3:
            continue
        boxes.append((x, y, ww, hh))

    boxes = sorted(boxes, key=lambda b: b[0])

    # Merge or prune boxes to target count 7-8
    if len(boxes) > 10:
        # remove very small widths
        mean_w = np.mean([b[2] for b in boxes])
        boxes = [b for b in boxes if b[2] > 0.5 * mean_w]
    # pad slightly and crop
    chars = []
    for (x, y, ww, hh) in boxes:
        pad_x = max(int(ww * 0.1), 2)
        pad_y = max(int(hh * 0.1), 2)
        x0 = max(x - pad_x, 0)
        y0 = max(y - pad_y, 0)
        x1 = min(x + ww + pad_x, w)
        y1 = min(y + hh + pad_y, h)
        crop = img[y0:y1, x0:x1]
        chars.append(crop)

    # heuristic: keep first 7-8 largest-height boxes
    if len(chars) >= 9:
        boxes_h = [(i, c.shape[0]) for i, c in enumerate(chars)]
        keep_idx = [i for i, _ in sorted(boxes_h, key=lambda x: x[1], reverse=True)[:8]]
        keep_idx = sorted(keep_idx)
        chars = [chars[i] for i in keep_idx]

    return chars


def preprocess_char_for_model(char_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    # Convert BGR -> RGB -> PIL -> transform
    rgb = cv2.cvtColor(char_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tfm = _build_transform(img_size)
    x = tfm(pil).unsqueeze(0)
    return x


def recognize_plate(image_path: str,
                     data_root: str,
                     chinese_model_path: str,
                     digits_model_path: str,
                     act_chinese: str,
                     act_digits: str,
                     img_size: int,
                     device: str) -> Tuple[str, List[Tuple[str, float]]]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    plate_img, _ = detect_plate(img)

    chars = segment_characters(plate_img)
    if not chars:
        # fallback: split evenly into 7 parts
        h, w = plate_img.shape[:2]
        splits = 7
        sub_w = w // splits
        chars = [plate_img[:, i * sub_w:(i + 1) * sub_w] for i in range(splits)]

    classes_ch = build_mapping_chinese(data_root)
    classes_dl = build_mapping_digits_letters(data_root)

    model_ch = load_model(chinese_model_path, num_classes=len(classes_ch), activation=act_chinese, device=device)
    model_dl = load_model(digits_model_path, num_classes=len(classes_dl), activation=act_digits, device=device)

    # ???? digits+letters ??????????????????? digits/letters ??????
    try:
        num_classes_dl = getattr(model_dl.classifier[-1], 'out_features')
    except Exception:
        num_classes_dl = None
    if num_classes_dl and num_classes_dl != len(classes_dl):
        num_train = os.path.join(data_root, 'dataset_num', 'train_28_28')
        num_test = os.path.join(data_root, 'dataset_num', 'testset_28_28')
        digits_classes = [str(i) for i in range(10) if os.path.isdir(os.path.join(num_train, str(i)))]
        union_letter_dirs = sorted(set(_list_classes(num_train)) | set(_list_classes(num_test)))
        letter_classes = [c for c in union_letter_dirs if _is_letter_class(c)]
        letters_sorted = sorted(letter_classes)
        digits_sorted = sorted(digits_classes)
        if num_classes_dl == len(digits_sorted):
            classes_dl = digits_sorted
        else:
            classes_dl = (letters_sorted + digits_sorted)[:num_classes_dl]

    preds = []
    plate_str = ''
    for idx, char_img in enumerate(chars):
        x = preprocess_char_for_model(char_img, img_size).to(device)
        with torch.no_grad():
            if idx == 0:
                logits = model_ch(x)
                prob = torch.softmax(logits, dim=1)
                pred_idx = int(prob.argmax(dim=1).item())
                conf = float(prob[0, pred_idx].item())
                pred = classes_ch[pred_idx] if 0 <= pred_idx < len(classes_ch) else str(pred_idx)
            else:
                logits = model_dl(x)
                prob = torch.softmax(logits, dim=1)
                pred_idx = int(prob.argmax(dim=1).item())
                conf = float(prob[0, pred_idx].item())
                pred = classes_dl[pred_idx] if 0 <= pred_idx < len(classes_dl) else str(pred_idx)
        preds.append((pred, conf))
        plate_str += pred

    return plate_str, preds


def main():
    parser = argparse.ArgumentParser(description='Recognize license plate from photo (detect, segment, classify)')
    parser.add_argument('--image_path', type=str, default=None, help='Single photo to process')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory of photos to batch process')
    parser.add_argument('--data_root', type=str, default=os.path.join(os.getcwd(), 'dataset'), help='Dataset root for class mapping')
    parser.add_argument('--chinese_model', type=str, default=os.path.join('outputs', 'model_20251025_164430.pt'), help='Path to Chinese model')
    parser.add_argument('--digits_model', type=str, default=os.path.join('outputs', 'model_20251025_163457.pt'), help='Path to digits+letters model')
    parser.add_argument('--act_chinese', choices=['relu', 'leaky_relu', 'elu'], default='elu', help='Activation used by Chinese model')
    parser.add_argument('--act_digits', choices=['relu', 'leaky_relu', 'elu'], default='relu', help='Activation used by digits+letters model')
    parser.add_argument('--img_size', type=int, default=28, help='Model input size for characters')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    photos = []
    if args.image_path:
        photos.append(args.image_path)
    if args.image_dir:
        for name in os.listdir(args.image_dir):
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                photos.append(os.path.join(args.image_dir, name))
    if not photos:
        raise ValueError('Provide --image_path or --image_dir with images')

    for p in sorted(photos):
        plate_str, details = recognize_plate(
            image_path=p,
            data_root=args.data_root,
            chinese_model_path=args.chinese_model,
            digits_model_path=args.digits_model,
            act_chinese=args.act_chinese,
            act_digits=args.act_digits,
            img_size=args.img_size,
            device=device,
        )
        print(f'Image: {p}')
        print(f'Recognized: {plate_str}')
        print('Characters: ' + ' '.join([f'{c}({conf:.2f})' for c, conf in details]))
        print('-' * 60)


if __name__ == '__main__':
    main()