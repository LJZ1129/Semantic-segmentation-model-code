# =================== train_deeplabv3plus.py ===================
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# 配置
DATA_ROOT = r"C:\Users\34114\Desktop\dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')

SAVE_DIR = Path(r"C:\Users\34114\Desktop\深度学习\CA第七代MobileNetV2")
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
CLASS_COLORS = [(255,255,255),(255,242,216),(255,255,217),(224,226,204),(255,152,153)]

# Coordinate Attention
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, 1, bias=False)
    def forward(self, x):
        n,c,h,w = x.size()
        x_h = F.adaptive_avg_pool2d(x,(h,1))
        x_w = F.adaptive_avg_pool2d(x,(1,w)).permute(0,1,3,2)
        y = torch.cat([x_h,x_w], 2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h,x_w = torch.split(y,[h,w],2)
        x_w = x_w.permute(0,1,3,2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

# DeepLabV3+ + CoordAtt
class DeepLabV3Plus_CA(nn.Module):
    def __init__(self, encoder_name='mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=5, ca_reduction=32, fallback_channels=256):
        super().__init__()
        self.base = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=classes, activation=None)
        self.encoder = self.base.encoder
        self.decoder = self.base.decoder
        self.segmentation_head = self.base.segmentation_head
        enc_channels = self.encoder.out_channels[-1]
        self.ca_enc = CoordAtt(enc_channels, ca_reduction)
        dec_channels = None
        if isinstance(self.segmentation_head, nn.Sequential) and len(self.segmentation_head)>0 and isinstance(self.segmentation_head[0], nn.Conv2d):
            dec_channels = self.segmentation_head[0].in_channels
        if dec_channels is None:
            dec_channels = fallback_channels
        self.ca_dec = CoordAtt(dec_channels, ca_reduction)
    def forward(self,x):
        feats = self.encoder(x)
        feats[-1] = self.ca_enc(feats[-1])
        try:
            dec = self.decoder(*feats)
        except TypeError:
            dec = self.decoder(feats)
        dec = self.ca_dec(dec)
        return self.segmentation_head(dec)

# 数据集 & 颜色映射
COLOR2ID = {(255,255,255):0,(255,242,216):1,(255,255,217):2,(224,226,204):3,(255,152,153):4}
def mask_rgb_to_id(mask_rgb):
    h,w,_ = mask_rgb.shape
    mask_id = np.zeros((h,w),np.uint8)
    for rgb,idx in COLOR2ID.items():
        loc = (mask_rgb[:,:,0]==rgb[0]) & (mask_rgb[:,:,1]==rgb[1]) & (mask_rgb[:,:,2]==rgb[2])
        mask_id[loc] = idx
    return mask_id

class RockDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, test_mode=False, test_ratio=0.2):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.names = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in ('.png','.tif','.tiff','.jpg')])
        if test_mode:
            k = max(1,int(len(self.names)*test_ratio))
            self.names = self.names[:k]
            print(f'[测试模式] 使用 {len(self.names)} 张样本')
        self.transform = transform
    def __len__(self): return len(self.names)
    def __getitem__(self,idx):
        name = self.names[idx]
        img_rgb = cv2.cvtColor(cv2.imread(str(self.img_dir/name)), cv2.COLOR_BGR2RGB)
        lbl_raw = cv2.imread(str(self.lbl_dir/name), cv2.IMREAD_UNCHANGED)
        if lbl_raw.ndim == 3:
            label = mask_rgb_to_id(cv2.cvtColor(lbl_raw, cv2.COLOR_BGR2RGB))
        else:
            label = lbl_raw.astype(np.int64)
        img_pil = Image.fromarray(img_rgb)
        label_pil = Image.fromarray(label.astype(np.uint8))
        if self.transform:
            return self.transform(img_pil, label_pil)
        else:
            img_tensor = T.ToTensor()(img_pil)
            img_tensor = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img_tensor)
            label_tensor = torch.from_numpy(np.array(label_pil)).long()
            return img_tensor, label_tensor

# ======= 无增强 Transform =======
class SimpleTransform:
    def __init__(self, resize=None):
        self.resize = resize
    def __call__(self, img_pil, mask_pil):
        if self.resize:
            img_pil = T.Resize(self.resize)(img_pil)
            mask_pil = T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)(mask_pil)
        img_tensor = T.ToTensor()(img_pil)
        img_tensor = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img_tensor)
        label_tensor = torch.from_numpy(np.array(mask_pil)).long()
        return img_tensor, label_tensor

# 评价指标
def calc_metrics(conf_mat, num_cls):
    ious, accs = [], []
    for i in range(num_cls):
        inter = conf_mat[i,i]
        union = conf_mat[i,:].sum() + conf_mat[:,i].sum() - inter
        total = conf_mat[i,:].sum()
        ious.append(inter/union if union>0 else 0.)
        accs.append(inter/total if total>0 else 0.)
    return np.mean(ious), ious, np.trace(conf_mat)/np.sum(conf_mat), np.mean(accs), accs

# 损失函数
class DiceSoftLoss(nn.Module):
    def __init__(self,smooth=1.): super().__init__(); self.smooth=smooth
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = torch.sum(probs*target_one_hot, dims)
        card = torch.sum(probs+target_one_hot, dims)
        dice = (2*inter + self.smooth)/(card + self.smooth)
        return 1 - dice.mean()
class MixedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceSoftLoss()
        self.w_ce, self.w_dice = ce_weight, dice_weight
    def forward(self,logits,target):
        return self.w_ce*self.ce(logits,target) + self.w_dice*self.dice(logits,target)

# Poly LR
def poly_lr(base_lr, cur_iter, max_iter, power=0.9):
    return base_lr*(1 - cur_iter/max_iter)**power

# 训练
def train_epoch(model, loader, criterion, optimizer, scaler, device, cur_iter, max_iter, base_lr, freeze_encoder):
    model.train()
    for p in model.encoder.parameters(): p.requires_grad_(not freeze_encoder)
    running_loss=0.
    for imgs, labels in tqdm(loader, desc='训练', leave=False):
        lr = poly_lr(base_lr, cur_iter, max_iter)
        for g in optimizer.param_groups: g['lr']=lr
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        cur_iter+=1
        running_loss += loss.item()*imgs.size(0)
    return running_loss/len(loader.dataset), cur_iter, lr

# 验证
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss=0.
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for imgs, labels in tqdm(loader, desc='验证', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item()*imgs.size(0)
        preds = torch.argmax(logits, 1).cpu().numpy()
        gts = labels.cpu().numpy()
        for lt, lp in zip(gts, preds):
            conf_mat += confusion_matrix(lt.flatten(), lp.flatten(), labels=list(range(NUM_CLASSES)))
    return running_loss/len(loader.dataset), *calc_metrics(conf_mat, NUM_CLASSES), conf_mat

# 混淆矩阵绘制
def plot_confusion_matrix(conf_mat, class_names, save_path):
    plt.figure(figsize=(8, 6))
    df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('预测类别'); plt.ylabel('真实类别'); plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 可视化结果
@torch.no_grad()
def visualize(model, loader, device, save_folder, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples==1: axes = axes.reshape(1,-1)
    mean = np.array([0.485,0.456,0.406]).reshape(1,1,3)
    std = np.array([0.229,0.224,0.225]).reshape(1,1,3)
    loader_iter = iter(loader)
    for i in range(num_samples):
        img_t, lbl_t = next(loader_iter)
        img_t = img_t.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = torch.argmax(model(img_t), 1)[0].cpu().numpy()
        img_np = img_t[0].cpu().permute(1,2,0).numpy()
        img_np = np.clip(img_np*std + mean,0,1)
        img_np = (img_np*255).astype(np.uint8)
        lbl_np = lbl_t[0].cpu().numpy()
        lbl_vis = np.zeros((*lbl_np.shape,3),np.uint8)
        prd_vis = np.zeros((*pred.shape,3),np.uint8)
        for idx, c in enumerate(CLASS_COLORS):
            lbl_vis[lbl_np==idx] = c
            prd_vis[pred==idx] = c
        axes[i,0].imshow(img_np); axes[i,0].set_title('原图'); axes[i,0].axis('off')
        axes[i,1].imshow(lbl_vis); axes[i,1].set_title('真值'); axes[i,1].axis('off')
        axes[i,2].imshow(prd_vis); axes[i,2].set_title('预测'); axes[i,2].axis('off')
    plt.tight_layout()
    plt.savefig(save_folder/'pred_vis.png', dpi=300)
    plt.close(fig)

# 主函数
def main():
    print("选择运行模式:\n1. 正常训练(100 epoch)\n2. 测试模式(样本1/5, 10 epoch)")
    mode = input("输入 1 或 2: ").strip()
    test_mode = (mode=='2')
    epochs = TEST_EPOCHS if test_mode else FULL_EPOCHS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    simple_transform = SimpleTransform(resize=PATCH_SIZE)

    train_ds = RockDataset(os.path.join(TRAIN_DIR, 'images'), os.path.join(TRAIN_DIR, 'labels'),
                           transform=simple_transform, test_mode=test_mode, test_ratio=0.2)
    val_ds = RockDataset(os.path.join(VAL_DIR, 'images'), os.path.join(VAL_DIR, 'labels'),
                         transform=simple_transform, test_mode=test_mode, test_ratio=0.2)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f'训练集样本数: {len(train_ds)}  |  验证集样本数: {len(val_ds)}')

    model = DeepLabV3Plus_CA(encoder_name='mobilenet_v2', encoder_weights='imagenet',
                             in_channels=3, classes=NUM_CLASSES, ca_reduction=32).to(device)
    criterion = MixedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=CLASS_WEIGHTS.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    best_miou = 0.0
    best_conf_mat = None
    g_iter = 0
    max_iters_total = epochs * len(train_ld)
    metrics_history = []

    for ep in range(1, epochs+1):
        freeze_flag = (ep <= FREEZE_EPS)
        tr_loss, g_iter, lr = train_epoch(model, train_ld, criterion, optimizer, scaler, device,
                                          g_iter, max_iters_total, BASE_LR, freeze_flag)
        vl_loss, miou, ious, oa, mean_acc, accs, conf_mat = validate_epoch(model, val_ld, criterion, device)
        miou_nbg = np.mean(ious[1:])
        mean_acc_nbg = np.mean(accs[1:])

        metrics_row = {
            'epoch': ep, 'train_loss': tr_loss, 'val_loss': vl_loss,
            'mIoU': miou, 'OA': oa, 'mean_acc': mean_acc,
            'mIoU_nbg': miou_nbg, 'mean_acc_nbg': mean_acc_nbg,
            **{f'IoU_{CLASS_NAMES[i]}': ious[i] for i in range(NUM_CLASSES)},
            **{f'Acc_{CLASS_NAMES[i]}': accs[i] for i in range(NUM_CLASSES)}
        }
        metrics_history.append(metrics_row)

        if miou > best_miou:
            best_miou, best_conf_mat = miou, conf_mat
            torch.save(model.state_dict(), SAVE_DIR / 'best_model.pth')
            print(f"[INFO] 新最佳模型保存于 Epoch {ep}，mIoU={miou:.4f}")

        print(f"Epoch {ep:3d}/{epochs}  LR={lr:.2e}  TrainLoss={tr_loss:.4f}  ValLoss={vl_loss:.4f}")
        print(f"  mIoU={miou:.4f}(最佳 {best_miou:.4f})  OA={oa:.4f}  MeanAcc={mean_acc:.4f}")
        print("  Cat-IoU: " + "  ".join(f"{n}:{v:.4f}" for n, v in zip(CLASS_NAMES, ious)))
        print("  Cat-Acc: " + "  ".join(f"{n}:{v:.4f}" for n, v in zip(CLASS_NAMES, accs)))
        print(f"  [忽略背景] mIoU={miou_nbg:.4f}  OA={oa:.4f}  MeanAcc={mean_acc_nbg:.4f}")
        print('-'*60)

    pd.DataFrame(metrics_history).to_csv(SAVE_DIR / 'metrics_history.csv', index=False, encoding='utf-8-sig')

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1), [m['train_loss'] for m in metrics_history], label='训练损失')
    plt.plot(range(1,epochs+1), [m['val_loss'] for m in metrics_history], label='验证损失')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('损失曲线'); plt.grid(); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,epochs+1), [m['mIoU'] for m in metrics_history], label='mIoU')
    plt.plot(range(1,epochs+1), [m['OA'] for m in metrics_history], label='OA')
    plt.plot(range(1,epochs+1), [m['mean_acc'] for m in metrics_history], label='MeanAcc')
    plt.xlabel('Epoch'); plt.ylabel('指标'); plt.title('验证指标'); plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR/'training_curves.png', dpi=300)
    plt.show()

    # 最佳模型可视化
    if (SAVE_DIR/'best_model.pth').exists():
        model.load_state_dict(torch.load(SAVE_DIR/'best_model.pth', map_location=device))
        if best_conf_mat is not None:
            plot_confusion_matrix(best_conf_mat, CLASS_NAMES, SAVE_DIR/'confusion_matrix.png')
        vis_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
        visualize(model, vis_loader, device, SAVE_DIR, num_samples=4)

if __name__ == '__main__':
    main()