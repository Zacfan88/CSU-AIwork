# -*- coding: utf-8 -*-

import argparse
import os

from data import get_digits_letters_datasets, get_chinese_datasets
from train import train_and_eval


def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition: CNN training & evaluation')
    parser.add_argument('--task', choices=['digits_letters', 'chinese'], default='digits_letters', help='Task: digits+letters or Chinese')
    parser.add_argument('--data_root', default=os.path.join(os.getcwd(), 'dataset'), help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu', help='Activation function')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--augment', action='store_true', help='Use light data augmentation for training set')
    args = parser.parse_args()

    if args.task == 'digits_letters':
        train_loader, test_loader, num_classes = get_digits_letters_datasets(args.data_root, img_size=args.img_size, batch_size=args.batch_size, augment=args.augment)
    else:
        train_loader, test_loader, num_classes = get_chinese_datasets(args.data_root, img_size=args.img_size, batch_size=args.batch_size, augment=args.augment)

    best_acc, model_path = train_and_eval(train_loader, test_loader, num_classes,
                                          epochs=args.epochs, lr=args.lr,
                                          optimizer_name=args.optimizer, activation=args.activation)
    print(f'Best test accuracy: {best_acc:.4f}')
    print(f'Model saved to: {model_path}')


if __name__ == '__main__':
    main()