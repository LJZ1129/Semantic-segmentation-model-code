# =================== 全部代码，从此行到末尾直接保存为 train_deeplabv1.py ===================
import os, random, warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 无法设置中文字体，请确保系统安装了对应字体")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# -- 全局随机种子 --
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -- 全局配置 --
DATA_ROOT = r"C:\Users\34114\Desktop\dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')

SAVE_DIR = Path(r"C:\Users\34114\Desktop\深度学习\DeepLabV1模型")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
BASE_LR = 1e-3
FULL_EPOCHS = 100
TEST_EPOCHS = 10
NUM_CLASSES = 5
PATCH_SIZE = (256, 256)
FREEZE_EPS = 5

CLASS_WEIGHTS = torch.tensor([0.6221, 0.3703, 1.0000, 0.3585, 0.2908], dtype=torch.float32)

CLASS_NAMES = ['背景', '砂岩', '第四纪沉积物', '大理岩', '闪长岩']
CLASS_COLORS = [(255, 255, 255), (255, 242, 216), (255, 255, 217), (224, 226, 204), (255, 152, 153)]

# -- 数据集工具 --
COLOR2ID = {tuple(k): i for i, k in enumerate(CLASS_COLORS)}
def mask_rgb_to_id(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    mask_id = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in COLOR2ID.items():
        loc = (mask_rgb[:, :, 0] == rgb[0]) & \
              (mask_rgb[:, :, 1] == rgb[1]) & \
              (mask_rgb[:, :, 2] == rgb[2])
        mask_id[loc] = idx
    return mask_id

class RockDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, test_mode=False, test_ratio=0.2):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.names = sorted([f.name for f in self.images_dir.iterdir()
                             if f.suffix.lower() in ('.png', '.tif', '.tiff', '.jpg')])
        if test_mode:
            k = max(1, int(len(self.names) * test_ratio))
            self.names = self.names[:k]
            print(f'[测试模式] 使用 {len(self.names)}/{len(os.listdir(self.images_dir))} 张样本')
        self.transform = transform
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = str(self.images_dir / name)
        label_path = str(self.labels_dir / name)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        label_raw = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_raw.ndim == 3:
            label_rgb = cv2.cvtColor(label_raw, cv2.COLOR_BGR2RGB)
            label_processed = mask_rgb_to_id(label_rgb)
        else:
            label_processed = label_raw.astype(np.int64)
        img_pil = Image.fromarray(img_rgb)
        label_pil = Image.fromarray(label_processed.astype(np.uint8))
        if self.transform:
            img_tensor, label_tensor = self.transform(img_pil, label_pil)
        else:
            img_tensor = T.ToTensor()(img_pil)
            img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
            label_tensor = torch.from_numpy(np.array(label_pil)).long()
        return img_tensor, label_tensor

class SegTransform:
    def __init__(self, resize=None, hflip_prob=0.5, vflip_prob=0.5, rotate_degree=15, color_jitter_params=None):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_degree = rotate_degree
        self.color_jitter_params = color_jitter_params if color_jitter_params else (0.2, 0.2, 0.2, 0.1)
    def __call__(self, img_pil, mask_pil):
        if self.resize:
            img_pil = T.Resize(self.resize)(img_pil)
            mask_pil = T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)(mask_pil)
        if random.random() < self.hflip_prob:
            img_pil = T.functional.hflip(img_pil)
            mask_pil = T.functional.hflip(mask_pil)
        if random.random() < self.vflip_prob:
            img_pil = T.functional.vflip(img_pil)
            mask_pil = T.functional.vflip(mask_pil)
        if self.rotate_degree > 0:
            angle = T.RandomRotation.get_params([-self.rotate_degree, self.rotate_degree])
            img_pil = T.functional.rotate(img_pil, angle)
            mask_pil = T.functional.rotate(mask_pil, angle, interpolation=T.InterpolationMode.NEAREST)
        img_pil = T.ColorJitter(*self.color_jitter_params)(img_pil)
        img_tensor = T.ToTensor()(img_pil)
        img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        label_tensor = torch.from_numpy(np.array(mask_pil)).long()
        return img_tensor, label_tensor

# -- DeepLabV1 实现 --
class DeepLabV1(nn.Module):
    def __init__(self, num_classes=21, in_channels=3):
        super(DeepLabV1, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())
        self.layer1 = nn.Sequential(*features[0:7])
        self.layer2 = nn.Sequential(*features[7:14])
        self.layer3 = nn.Sequential(*features[14:24])
        self.layer4 = nn.Sequential(*features[24:34])
        for m in self.layer4:
            if isinstance(m, nn.Conv2d):
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
        self.aspp = nn.ModuleList()
        rates = [6, 12, 18, 24]
        in_ch = 512
        for r in rates:
            self.aspp.append(nn.Conv2d(in_ch, 512, kernel_size=3, padding=r, dilation=r))
        self.classifier = nn.Conv2d(512 * len(rates), num_classes, kernel_size=1)
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        aspp_outs = []
        for conv in self.aspp:
            aspp_outs.append(conv(x))
        x = torch.cat(aspp_outs, dim=1)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

# -- 评价指标 --
def calc_metrics(conf_mat, num_cls):
    ious, accs = [], []
    for i in range(num_cls):
        inter = conf_mat[i, i]
        union = conf_mat[i, :].sum() + conf_mat[:, i].sum() - inter
        total_gt_pixels = conf_mat[i, :].sum()
        ious.append(inter / union if union > 0 else 0.)
        accs.append(inter / total_gt_pixels if total_gt_pixels > 0 else 0.)
    miou = np.mean(ious)
    mean_acc = np.mean(accs)
    oa = np.trace(conf_mat) / np.sum(conf_mat) if conf_mat.sum() > 0 else 0.
    return miou, ious, oa, mean_acc, accs

# -- 损失函数 --
class DiceSoftLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class MixedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceSoftLoss()
        self.w_ce, self.w_dice = ce_weight, dice_weight
    def forward(self, logits, target):
        return self.w_ce * self.ce(logits, target) + self.w_dice * self.dice(logits, target)

# -- Poly LR --
def poly_lr(base_lr, cur_iter, max_iter, power=0.9):
    return base_lr * (1 - cur_iter / max_iter) ** power

# -- 训练 --
def train_epoch(model, loader, criterion, optimizer, scaler, device,
                cur_iter, max_iter, base_lr, freeze_encoder):
    model.train()
    if hasattr(model, 'layer1'): # freeze前端特征
        for p in model.layer1.parameters():
            p.requires_grad_(not freeze_encoder)
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc='训练', leave=False):
        current_lr = poly_lr(base_lr, cur_iter, max_iter)
        for g in optimizer.param_groups:
            g['lr'] = current_lr
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        cur_iter += 1
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset), cur_iter, current_lr

# -- 验证 --
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for imgs, labels in tqdm(loader, desc='验证', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts = labels.cpu().numpy()
        for lt, lp in zip(gts, preds):
            cm = confusion_matrix(lt.flatten(), lp.flatten(), labels=list(range(NUM_CLASSES)))
            conf_mat += cm
    miou, ious, oa, mean_acc, accs = calc_metrics(conf_mat, NUM_CLASSES)
    return running_loss / len(loader.dataset), miou, ious, oa, mean_acc, accs, conf_mat

# -- 绘制混淆矩阵 --
def plot_confusion_matrix(conf_mat, class_names, save_path):
    plt.figure(figsize=(8, 6))
    df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 10})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('混淆矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# -- 可视化 --
@torch.no_grad()
def visualize(model, loader, device, save_folder, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    loader_iter = iter(loader)
    for i in range(num_samples):
        try:
            img_t, lbl_t = next(loader_iter)
        except StopIteration:
            break
        img_t = img_t.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(img_t)
        pred = torch.argmax(logits, 1)[0].cpu().numpy()
        img_np = img_t[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        lbl_np = lbl_t[0].cpu().numpy()
        lbl_vis = np.zeros((*lbl_np.shape, 3), np.uint8)
        prd_vis = np.zeros((*pred.shape, 3), np.uint8)
        for idx, c in enumerate(CLASS_COLORS):
            lbl_vis[lbl_np == idx] = c
            prd_vis[pred == idx] = c
        axes[i, 0].imshow(img_np); axes[i, 0].set_title('原图'); axes[i, 0].axis('off')
        axes[i, 1].imshow(lbl_vis); axes[i, 1].set_title('真值'); axes[i, 1].axis('off')
        axes[i, 2].imshow(prd_vis); axes[i, 2].set_title('预测'); axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_folder / 'pred_vis.png', dpi=300)
    plt.close(fig)

# -- 主训练入口 --
def main():
    print("选择运行模式:\n1. 正常训练(100 epoch)\n2. 测试模式(样本1/5, 10 epoch)")
    mode = input("输入 1 或 2: ").strip()
    test_mode = (mode == '2')
    epochs = TEST_EPOCHS if test_mode else FULL_EPOCHS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    transform = SegTransform(resize=PATCH_SIZE, hflip_prob=0.5, vflip_prob=0.5,
                             rotate_degree=15, color_jitter_params=(0.2, 0.2, 0.2, 0.1))
    val_transform = SegTransform(resize=PATCH_SIZE, hflip_prob=0.0, vflip_prob=0.0,
                                rotate_degree=0, color_jitter_params=(0, 0, 0, 0))

    persistent_workers = True if torch.cuda.is_available() and torch.backends.cudnn.enabled else False
    train_ds = RockDataset(os.path.join(TRAIN_DIR, 'images'), os.path.join(TRAIN_DIR, 'labels'),
                           transform=transform, test_mode=test_mode, test_ratio=0.2)
    val_ds = RockDataset(os.path.join(VAL_DIR, 'images'), os.path.join(VAL_DIR, 'labels'),
                         transform=val_transform, test_mode=test_mode, test_ratio=0.2)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=persistent_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=persistent_workers)
    print(f'训练集样本数: {len(train_ds)}  |  验证集样本数: {len(val_ds)}')

    # 使用 DeepLabV1 模型
    model = DeepLabV1(num_classes=NUM_CLASSES, in_channels=3).to(device)

    criterion = MixedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=CLASS_WEIGHTS.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    best_miou, best_conf_mat = 0.0, None
    g_iter, max_iters_total = 0, epochs * len(train_ld)
    train_losses, val_losses, val_mious, val_oas, val_meanaccs = [], [], [], [], []
    metrics_history = []

    for ep in range(1, epochs + 1):
        freeze_encoder_flag = (ep <= FREEZE_EPS)
        tr_loss, g_iter, current_lr = train_epoch(model, train_ld, criterion, optimizer, scaler, device,
                                                  g_iter, max_iters_total, BASE_LR, freeze_encoder_flag)
        vl_loss, miou, ious, oa, mean_acc_val, accs_val, conf_mat_curr = validate_epoch(model, val_ld, criterion, device)
        ious_nbg = ious[1:]
        accs_nbg = accs_val[1:]
        miou_nbg = np.mean(ious_nbg)
        mean_acc_nbg = np.mean(accs_nbg)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_mious.append(miou)
        val_oas.append(oa)
        val_meanaccs.append(mean_acc_val)

        row = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': vl_loss, 'mIoU': miou,
               'OA': oa, 'mean_acc': mean_acc_val, 'mIoU_nobg': miou_nbg, 'mean_acc_nobg': mean_acc_nbg}
        for idx, cname in enumerate(CLASS_NAMES):
            row[f'IoU_{cname}'] = ious[idx]
        for idx, cname in enumerate(CLASS_NAMES):
            row[f'Acc_{cname}'] = accs_val[idx]
        metrics_history.append(row)

        if miou > best_miou:
            best_miou, best_conf_mat = miou, conf_mat_curr
            torch.save(model.state_dict(), SAVE_DIR / 'best_model.pth')
            print(f"[INFO] 新最佳模型保存于 Epoch {ep}，mIoU={miou:.4f}")

        print(f"Epoch {ep}/{epochs} LR={current_lr:.2e} TrainLoss={tr_loss:.4f} ValLoss={vl_loss:.4f}")
        print(f"  mIoU={miou:.4f}(最佳 {best_miou:.4f}) OA={oa:.4f} MeanAcc={mean_acc_val:.4f}")
        print("  Cat-IoU: " + "  ".join(f"{n}:{v:.4f}" for n, v in zip(CLASS_NAMES, ious)))
        print("  Cat-Acc: " + "  ".join(f"{n}:{v:.4f}" for n, v in zip(CLASS_NAMES, accs_val)))
        print(f"  [忽略背景] mIoU={miou_nbg:.4f} MeanAcc={mean_acc_nbg:.4f}")
        print('-'*60)

        torch.cuda.empty_cache()

    pd.DataFrame(metrics_history).to_csv(SAVE_DIR / 'metrics_history.csv', index=False, encoding='utf-8-sig')
    print(f"[INFO] 已保存每个 epoch 的指标到: {SAVE_DIR / 'metrics_history.csv'}")

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, epochs + 1), val_losses, label='验证损失')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('损失曲线'); plt.grid(); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_mious, label='mIoU')
    plt.plot(range(1, epochs + 1), val_oas, label='OA')
    plt.plot(range(1, epochs + 1), val_meanaccs, label='平均精度')
    plt.xlabel('Epoch'); plt.ylabel('指标'); plt.title('验证指标'); plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'training_curves.png', dpi=300)
    plt.show()

    if (SAVE_DIR / 'best_model.pth').exists():
        model.load_state_dict(torch.load(str(SAVE_DIR / 'best_model.pth'), map_location=device))
        if best_conf_mat is not None:
            plot_confusion_matrix(best_conf_mat, CLASS_NAMES, SAVE_DIR / 'confusion_matrix.png')
            print(f"[INFO] 混淆矩阵图保存于: {SAVE_DIR / 'confusion_matrix.png'}")
        vis_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
        visualize(model, vis_loader, device, SAVE_DIR, num_samples=4)
        print(f"[INFO] 预测结果可视化图保存于: {SAVE_DIR / 'pred_vis.png'}")

if __name__ == '__main__':
    main()