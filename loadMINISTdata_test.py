import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np

# =================================数据预处理=================================
transform = transforms.Compose([
    transforms.ToTensor(),
])

# =================================加载全部数据=================================
print("正在加载MNIST数据集...")
full_train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# =================================按类别划分=================================
def split_by_label(dataset, target_label=2):
    """根据标签划分数据，返回(目标类别数据, 其他类别数据)"""
    target_indices = []
    other_indices = []
    
    for idx, (_, label) in enumerate(dataset):
        if label == target_label:
            target_indices.append(idx)      # 类别2的索引
        else:
            other_indices.append(idx)       # 非类别2的索引
    
    target_data = Subset(dataset, target_indices)   # 类别2的数据
    other_data = Subset(dataset, other_indices)     # 非类别2的数据
    
    return target_data, other_data

# 划分数据
print("按类别划分数据...")
# 修正：other_data 是非2的数据（训练集用）
# target_data 是2的数据（测试集用）
test_data_raw, train_data_raw = split_by_label(full_train_dataset, target_label=2)

print(f"训练数据（除2外）: {len(train_data_raw)} 张")
print(f"测试数据（类别2）: {len(test_data_raw)} 张")

# =================================划分验证集=================================
# 从训练数据（不含2）中取20%作为验证集
val_size = int(0.2 * len(train_data_raw))
train_size = len(train_data_raw) - val_size

train_dataset, val_dataset = random_split(
    train_data_raw, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"训练集: {len(train_dataset)} 张 (不含2)")
print(f"验证集: {len(val_dataset)} 张 (不含2)")
print(f"测试集: {len(test_data_raw)} 张 (全是2)")

# =================================创建DataLoader=================================
batch_size = 64

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

valid_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_data_raw, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =================================验证数据分布=================================
def verify_distribution(loader, name, should_contain_2=False):
    """验证数据分布"""
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy().tolist())
    
    unique_labels = np.unique(labels)
    print(f"\n{name}:")
    print(f"  样本数: {len(labels)}")
    print(f"  包含的类别: {unique_labels}")
    
    has_2 = 2 in unique_labels
    if should_contain_2:
        print(f"  包含类别2: {has_2} " if has_2 else "  包含类别2: False ")
    else:
        print(f"  包含类别2: {has_2} " if has_2 else "  包含类别2: False ")
    
    return labels

print("\n" + "="*50)
print("数据分布验证")
print("="*50)

# 验证各个数据集
train_labels = verify_distribution(train_loader, "训练集", should_contain_2=False)
valid_labels = verify_distribution(valid_loader, "验证集", should_contain_2=False)
test_labels = verify_distribution(test_loader, "测试集", should_contain_2=True)

# 详细标签分布
print("\n" + "="*50)
print("详细标签分布")
print("="*50)

for name, loader in [("训练集", train_loader), ("验证集", valid_loader)]:
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy().tolist())
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} (不含2):")
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 张 ({count/len(labels)*100:.1f}%)")

# 测试集详细分布
labels = []
for _, lbl in test_loader:
    labels.extend(lbl.numpy().tolist())
unique, counts = np.unique(labels, return_counts=True)
print(f"\n测试集 (全是2):")
for label, count in zip(unique, counts):
    print(f"  类别 {label}: {count} 张 ({count/len(labels)*100:.1f}%)")

print("\n数据加载完成！")