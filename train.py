import argparse
import torch
from torch.nn import functional as F
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import time
from torchvision.models import wide_resnet50_2
from torchvision.models.resnet import Bottleneck
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
import pytorch_lightning as pl
import string
import random

# Additional imports for AU-PRO calculation
from bisect import bisect
from scipy.ndimage import label  # for connected component analysis

# -----------------------------
# Helper functions
# -----------------------------
def prep_dirs(root):
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    return sample_path

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def auto_select_weights_file(weights_file_version):
    print()
    version_list = glob.glob(os.path.join(args.project_path, args.category) + '/lightning_logs/version_*')
    version_list.sort(reverse=True, key=lambda x: os.path.getmtime(x))
    if weights_file_version is not None:
        version_list = [os.path.join(args.project_path, args.category) + '/lightning_logs/' + weights_file_version] + version_list
    for i in range(len(version_list)):
        weights_file_path = glob.glob(os.path.join(version_list[i],'checkpoints')+'/*')
        if len(weights_file_path) == 0:
            if weights_file_version is not None and i == 0:
                print(f'Checkpoint of {weights_file_version} not found')
            continue
        else:
            weights_file_path = weights_file_path[0]
            if weights_file_path.split('.')[-1] != 'ckpt':
                continue
        print('Checkpoint found : ', weights_file_path)
        print()
        return weights_file_path
    print('Checkpoint not found')
    print()
    return None

# Imagenet normalization constants
mean_train = [0.485, 0.456, 0.406]
std_train  = [0.229, 0.224, 0.225]

# -----------------------------
# Loss Functions
# -----------------------------
def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    # inputs are assumed to be probabilities (after sigmoid)
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()

def tversky_loss(inputs, targets, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss:
      Tversky index = (TP + smooth) / (TP + α·FP + β·FN + smooth)
      Loss = 1 – index
    """
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    return 1.0 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)


# -----------------------------
# AU-PRO Utility Functions 
# -----------------------------
class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    """
    def __init__(self, anomaly_scores):
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)

def trapezoid(x, y, x_max=None):
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Some values are not finite. Continuing with finite values only.")
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as anomaly scores for all potential false positive pixels.
    """
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)
    structure = np.ones((3, 3), dtype=int)
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        # Ensure ground truth is 2D by squeezing extra dimensions.
        gt_map = np.squeeze(gt_map)
        labeled, n_components = label(gt_map, structure)
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index : ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()
    return ground_truth_components, anomaly_scores_ok_pixels

def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)
    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)
        fprs.append(fpr)
        pros.append(pro)
    fprs = fprs[::-1]
    pros = pros[::-1]
    return fprs, pros

def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    return au_pro, pro_curve

# -----------------------------
# Quantum Block
# -----------------------------
class QuantumBlock(nn.Module):
    """Wrap a ResNet Bottleneck to insert two simple quantum-inspired rotations."""
    def __init__(self, orig_block: Bottleneck):
        super().__init__()
        # copy over all original layers
        self.conv1, self.bn1 = orig_block.conv1, orig_block.bn1
        self.conv2, self.bn2 = orig_block.conv2, orig_block.bn2
        self.conv3, self.bn3 = orig_block.conv3, orig_block.bn3
        self.relu = orig_block.relu
        self.downsample = orig_block.downsample
        # two learnable “rotation angles” per block
        self.theta1 = nn.Parameter(torch.randn(self.conv1.out_channels))
        self.theta2 = nn.Parameter(torch.randn(self.conv2.out_channels))

    def forward(self, x):
        identity = x

        # first conv + quantum-inspired rotation
        out = self.conv1(x)
        # rotate each channel:  out → sin(out)*cos(θ) + cos(out)*sin(θ)
        t1 = self.theta1.view(1, -1, 1, 1)
        out = torch.sin(out) * torch.cos(t1) + torch.cos(out) * torch.sin(t1)
        out = self.bn1(out)
        out = self.relu(out)

        # second conv + another rotation
        out = self.conv2(out)
        t2 = self.theta2.view(1, -1, 1, 1)
        out = torch.sin(out) * torch.cos(t2) + torch.cos(out) * torch.sin(t2)
        out = self.bn2(out)
        out = self.relu(out)

        # third conv (no rotation here)
        out = self.conv3(out)
        out = self.bn3(out)

        # downsample if needed, then add skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class QuantumWideResNet50_2(nn.Module):
    """A WideResNet-50_2 where each Bottleneck is replaced by QuantumBlock."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # load the standard WideResNet50_2
        base = wide_resnet50_2(pretrained=pretrained)
        self.backbone = base

        # expose the same API as a normal resnet so that
        # `model.layer1` still works:
        for name in (
            'conv1','bn1','relu','maxpool',
            'layer1','layer2','layer3','layer4',
            'avgpool','fc'
        ):
            setattr(self, name, getattr(self.backbone, name))
 
    def forward(self, x):
        return self.backbone(x)

# -----------------------------
# Discriminator Definition
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=2048, feature_map_size=8):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.fc1   = nn.Linear(256 * feature_map_size * feature_map_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# -----------------------------
# Dataset Definition
# -----------------------------
class MedAD_Dataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path  = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths  = []
        tot_labels   = []
        tot_types    = []
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths  = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Mismatch between test images and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path, gt_item, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt_item == 0:
            gt_tensor = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            gt_img = Image.open(gt_item)
            gt_tensor = self.gt_transform(gt_img)
            gt_tensor = (gt_tensor > 0.5).float()  # binarize
        assert img.size()[1:] == gt_tensor.size()[1:], "Image and ground truth dimensions do not match!"
        return img, gt_tensor, label, os.path.basename(img_path[:-4]), img_type

# -----------------------------
# Helper Functions for Visualization
# -----------------------------
def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])
    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive', false_p)
    print('false negative', false_n)

# -----------------------------
# Improved Segmentation Head (Decoder)
# -----------------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(SegmentationHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, student_features, anomaly_map):
        x = torch.cat([student_features, anomaly_map], dim=1)
        x = self.decoder(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return x

# -----------------------------
# Main Model: MED_AD with Two-Stage Segmentation Refinement
# -----------------------------
class MED_AD(pl.LightningModule):
    def __init__(self, hparams):
        super(MED_AD, self).__init__()
        self.save_hyperparameters(hparams)
        self.init_features()
        
        # Hook functions to capture features from teacher and student
        def hook_t(module, input, output):
            self.features_t.append(output)
        def hook_s(module, input, output):
            self.features_s.append(output)
        
        # teacher: quantum-inspired WideResNet-50_2
        self.model_t = QuantumWideResNet50_2(pretrained=True).eval()
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)
        self.model_t.layer4[-1].register_forward_hook(hook_t)
        
        # student: same quantum-inspired backbone
        self.model_s = QuantumWideResNet50_2(pretrained=False)
        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)
        self.model_s.layer4[-1].register_forward_hook(hook_s)
        
        self.discriminator = Discriminator(input_size=2048, feature_map_size=8)
        self.criterion_g = torch.nn.MSELoss(reduction='sum')
        self.criterion_d = torch.nn.BCELoss()
        
        self.student_seg = SegmentationHead(in_channels=2048, num_classes=1)
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.inference_times = []
        # For AU-PRO and dice score metrics:
        self.au_pro_gt_maps = []
        self.au_pro_pred_maps = []
        self.dice_scores = []

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.hparams['load_size'], self.hparams['load_size']), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(self.hparams['input_size']),
            transforms.Normalize(mean=mean_train, std=std_train)
        ])

        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.hparams['load_size'], self.hparams['load_size'])),
            transforms.ToTensor(),
            transforms.CenterCrop(self.hparams['input_size'])
        ])
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    
    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.au_pro_gt_maps = []
        self.au_pro_pred_maps = []
        self.dice_scores = []
    
    def init_features(self):
        self.features_t = []
        self.features_s = []
    
    def forward(self, x):
        self.init_features()
        _ = self.model_t(x)
        _ = self.model_s(x)
        return self.features_t, self.features_s
    
    def cal_loss(self, fs_list, ft_list):
        tot_loss = 0
        for fs, ft in zip(fs_list, ft_list):
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            f_loss = (0.5 / (fs.shape[2] * fs.shape[3])) * self.criterion_g(fs_norm, ft_norm)
            tot_loss += f_loss
        return tot_loss
    
    def cal_anomaly_map(self, fs_list, ft_list, out_size=224):
        if self.hparams.amap_mode == 'mul':
            anomaly_map = np.ones([out_size, out_size])
        else:
            anomaly_map = np.zeros([out_size, out_size])
        a_map_list = []
        for fs, ft in zip(fs_list, ft_list):
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')[0, 0, :, :].cpu().detach().numpy()
            a_map_list.append(a_map)
            if self.hparams.amap_mode == 'mul':
                anomaly_map *= a_map
            else:
                anomaly_map += a_map
        anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0.5)
        return anomaly_map, a_map_list
    
    def save_anomaly_map(self, anomaly_map, a_maps, input_img, gt_img, file_name, x_type):
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)
        am64 = cvt2heatmap(min_max_norm(a_maps[0]) * 255)
        am32 = cvt2heatmap(min_max_norm(a_maps[1]) * 255)
        am16 = cvt2heatmap(min_max_norm(a_maps[2]) * 255)
        am8  = cvt2heatmap(min_max_norm(a_maps[3]) * 255)
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_m64.jpg'), am64)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_m32.jpg'), am32)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_m16.jpg'), am16)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_m8.jpg'), am8)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_map.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_map_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)
        seg_img = np.uint8((anomaly_map_norm >= 0.5) * 255)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_seg.jpg'), seg_img)
    
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            list(self.model_s.parameters()) + list(self.student_seg.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return [optimizer_g, optimizer_d], []
    
    def train_dataloader(self):
        image_datasets = MedAD_Dataset(
            root=os.path.join(self.hparams.dataset_path, self.hparams.category),
            transform=self.data_transforms,
            gt_transform=self.gt_transforms,
            phase='train'
        )
        train_loader = DataLoader(image_datasets, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)
        return train_loader
    
    def test_dataloader(self):
        test_datasets = MedAD_Dataset(
            root=os.path.join(self.hparams.dataset_path, self.hparams.category),
            transform=self.data_transforms,
            gt_transform=self.gt_transforms,
            phase='test'
        )
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader
    
    def on_train_start(self):
        self.model_t.eval()
        self.sample_path = prep_dirs(self.logger.log_dir)
    
    def on_test_start(self):
        self.init_results_list()
        self.sample_path = prep_dirs(self.logger.log_dir)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, gt, label, file_name, _ = batch
        features_t, features_s = self(x)
        if optimizer_idx == 0:  # Generator step
            features_t, features_s = self(x)
            loss_g = self.cal_loss(features_s, features_t)

            # always build the anomaly_map and segmentation prediction:
            fs = features_s[-1]
            ft = features_t[-1]
            target_size = ft.shape[2:]
            fs_resized = F.adaptive_avg_pool2d(fs, target_size)
            fs_norm = F.normalize(fs_resized, p=2)
            ft_norm = F.normalize(F.adaptive_avg_pool2d(ft, target_size), p=2)
            anomaly_map_t = 1 - F.cosine_similarity(fs_norm, ft_norm, dim=1, eps=1e-6)
            anomaly_map_t = anomaly_map_t.unsqueeze(1)
            seg_pred = self.student_seg(fs_resized, anomaly_map_t)

            # split out positive vs. negative images
            mask = label.view(-1) == 1
            if mask.any():
                # positive images: use their real gt for seg loss
                gt_pos = gt[mask]
                pred_pos = seg_pred[mask]
                tver_pos = tversky_loss(pred_pos, gt_pos, alpha=0.7, beta=0.3)
                seg_loss_pos = focal_loss(pred_pos, gt_pos) + dice_loss(pred_pos, gt_pos) + self.hparams.lambda_tv * tver_pos
            else:
                seg_loss_pos = 0.0

            if (~mask).any():
                # negative images: gt mask is all zeros
                pred_neg = seg_pred[~mask]
                zeros = torch.zeros_like(pred_neg)
                zeros = torch.zeros_like(pred_neg)
                tver_neg = tversky_loss(pred_neg, zeros, alpha=0.7, beta=0.3)
                seg_loss_neg = focal_loss(pred_neg, zeros) + dice_loss(pred_neg, zeros) + self.hparams.lambda_tv * tver_neg
            else:
                seg_loss_neg = 0.0

            seg_loss = seg_loss_pos + seg_loss_neg
            total_loss = loss_g + self.hparams.lambda_seg * seg_loss
            self.log('train_seg_loss', seg_loss, on_epoch=True)
            self.log('train_loss_g', loss_g, on_epoch=True)
            return total_loss

        if optimizer_idx == 1:  # Discriminator step
            real_labels = torch.ones((x.size(0), 1), device=self.device)
            fake_labels = torch.zeros((x.size(0), 1), device=self.device)
            real_loss = self.criterion_d(self.discriminator(features_t[-1]), real_labels)
            fake_loss = self.criterion_d(self.discriminator(features_s[-1]), fake_labels)
            loss_d = real_loss + fake_loss
            self.log('train_loss_d', loss_d, on_epoch=True)
            return loss_d
    
    def test_step(self, batch, batch_idx):
        start_time = time.time()
        x, gt, label, file_name, x_type = batch
        features_t, features_s = self(x)
        
        anomaly_map, a_map_list = self.cal_anomaly_map(features_s, features_t, out_size=self.hparams.input_size)
        gt_np = gt.cpu().numpy().astype(int)
        
        # Extend pixel-level and image-level lists
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(anomaly_map.max())
        self.img_path_list.extend(file_name)
        
        # Store anomaly maps and corresponding ground truth for AU-PRO calculation.
        self.au_pro_pred_maps.append(anomaly_map)
        # Append the full ground truth mask instead of a scalar value:
        self.au_pro_gt_maps.append(gt_np[0])
        
        # Generate seg_img from the normalized anomaly map
        anomaly_map_norm = min_max_norm(anomaly_map)
        seg_img_from_map = np.uint8((anomaly_map_norm >= 0.5) * 255)
        
        # (If needed, you can calculate Dice score here using seg_img_from_map vs. gt_np[0])
        if label[0].item() == 1:
             #cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_seg.jpg'), seg_img_from_map)
             pred_mask = seg_img_from_map / 255.0
             gt_mask = gt_np[0]  # full 2D ground truth mask
             smooth = 1.0
             intersection = np.sum(pred_mask * gt_mask)
             dice = (2.0 * intersection + smooth) / (np.sum(pred_mask) + np.sum(gt_mask) + smooth)
             self.dice_scores.append(dice)
        
        x_inv = self.inv_normalize(x)
        input_x = cv2.cvtColor(x_inv.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map, a_map_list, input_x, gt_np[0][0] * 255, file_name[0], x_type[0])
        
        end_time = time.time()
        self.inference_times.append(end_time - start_time)

    
    def test_epoch_end(self, outputs):
        print("Total pixel-level AUC-ROC score:")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level AUC-ROC score:")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        avg_inference_time = np.mean(self.inference_times)
        print(f"Average inference time per image: {avg_inference_time} seconds")
        self.log('avg_inference_time', avg_inference_time)
        
        # Compute and log average Dice score (only computed on abnormal images)
        if len(self.dice_scores) > 0:
            avg_dice = np.mean(self.dice_scores)
        else:
            avg_dice = 0.0
        print("Average Dice Score:", avg_dice)
        self.log('avg_dice', avg_dice)
        
        # Compute and log AU-PRO score.
        au_pro, pro_curve = calculate_au_pro(self.au_pro_gt_maps, self.au_pro_pred_maps, integration_limit=0.3, num_thresholds=100)
        print("AU-PRO:", au_pro)
        self.log('au_pro', au_pro)

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./Med_AD')
    parser.add_argument('--category', default='brain')
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_size', default=256, type=int)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--project_path', default=r'./Med_Ad')
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul','sum'], default='mul')
    parser.add_argument('--weights_file_version', type=str, default=None)
    parser.add_argument('--lambda_seg', default=1.0, type=float)
    parser.add_argument('--lambda_tv', type=float, default=0.6, help='weight for Tversky loss in segmentation head')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=os.path.join(args.project_path, args.category),
        max_epochs=args.num_epochs,
        gpus=[0] if torch.cuda.is_available() else None
    )
    
    if args.phase == 'train':
        model = MED_AD(hparams=args)
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        weights_file_path = auto_select_weights_file(args.weights_file_version)
        if weights_file_path is not None:
            model = MED_AD(hparams=args).load_from_checkpoint(weights_file_path)
            trainer.test(model)
        else:
            print('Weights file is not found!')

