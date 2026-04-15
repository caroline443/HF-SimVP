# verify_data.py
from dataset_universal import SEVIRDataset

root = "F:\zyx\HF-SimVP\dataset\sevir_data"

print("正在检查训练集...")
try:
    train_ds = SEVIRDataset(root, mode='train')
    print(f"训练集样本数: {len(train_ds)}") # 应该有好几万
except Exception as e:
    print(f"训练集加载失败: {e}")

print("\n正在检查测试集...")
try:
    test_ds = SEVIRDataset(root, mode='test')
    print(f"测试集样本数: {len(test_ds)}") # 应该也有不少
except Exception as e:
    print(f"测试集加载失败: {e}")