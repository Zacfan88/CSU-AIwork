# 车牌识别实验（三）

本项目实现两个卷积神经网络（CNN）：
- 数字+字母识别（24个字母，排除 I/O；10 个数字）
- 汉字识别（31 个省份/地区简称）

数据目录（默认）：`dataset/`
- `dataset/dataset_num/train_28_28`, `dataset/dataset_num/testset_28_28`（0-9 数字）
- `dataset/dataset_char/train_28_28`, `dataset/dataset_char/test_28_28`（字母与汉字）

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 运行示例

数字+字母识别：
```bash
python3 src/main.py --task digits_letters --epochs 10 --lr 1e-3 --batch_size 64 --activation relu --optimizer adam
```

汉字识别：
```bash
python3 src/main.py --task chinese --epochs 10 --lr 1e-3 --batch_size 64 --activation relu --optimizer sgd
```

如果你的数据目录不在默认位置，可显式指定：
```bash
python3 src/main.py --task digits_letters --data_root "/Users/fan/Downloads/ailab3/dataset" --epochs 10 --lr 1e-3
```

## 可调参数
- `--epochs` 训练轮数
- `--lr` 学习率
- `--batch_size` 批大小
- `--activation` 激活函数：`relu` / `leaky_relu` / `elu`
- `--optimizer` 优化器：`adam` / `sgd`
- `--img_size` 输入图像大小（默认 28）

## 说明
- 数字+字母任务将 `dataset_char` 中的字母类（A-H, J-N, P-Z）与 `dataset_num` 中的数字类（0-9）拼接为一个数据集训练。
- 汉字任务仅使用 `dataset_char` 中的汉字类别（通过 Unicode 范围判断）训练。
- 模型与训练日志会保存到 `outputs/` 目录。