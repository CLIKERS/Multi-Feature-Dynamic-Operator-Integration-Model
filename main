import math
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMRegressor
import torch.optim as optim
import numpy as np
import warnings
import autosklearn.regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
# 设置警告过滤器，忽略特定类型的警告
warnings.filterwarnings('ignore', message="X does not have valid feature names, but RandomForestRegressor was fitted with feature names")

df = pd.read_excel("data2.xlsx")
print(df.head(10))
feature = df.iloc[:,2:11]
print(feature.head(10))
lable = df.iloc[:,11:]
print(lable.head(10))
selected_columns = ['GR', 'CALI', 'SP', 'RT', 'RI', 'RXO', 'DEN', 'AC', 'CNL', 'Clay', 'Cal', 'Dol', 'K_f', 'Na_f', 'Pyrite', 'Quartz']
df_selected = df[selected_columns]

# 计算相关性矩阵
correlation_matrix = df_selected.corr()
selected_correlation_matrix = correlation_matrix.loc[['GR', 'CALI', 'SP', 'RT', 'RI', 'RXO', 'DEN', 'AC', 'CNL'], ['Clay', 'Cal', 'Dol', 'K_f', 'Na_f', 'Pyrite', 'Quartz']]

# 绘制热力图
plt.figure(figsize=(12, 10))
ax = sns.heatmap(selected_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")

# sns.heatmap(selected_correlation_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 14})
plt.xticks(fontsize=14,fontname='Times New Roman')  # 设置横坐标标签字体大小
plt.yticks(fontsize=14,fontname='Times New Roman')  # 设置纵坐标标签字体大小

# 遍历并设置注释字体
for text in ax.texts:
    text.set_fontname('Times New Roman')  # 设置注释字体为Times New Roman
    text.set_fontsize(16)  # 可选：设置注释字体大小

# 获取色条对象并设置其字体属性
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)  # 设置色条刻度标签的大小（可选）
for t in cbar.ax.get_yticklabels():
    t.set_fontname('Times New Roman')  # 设置色条刻度标签的字体为Times New Roman
plt.show()


# 定义自定义的损失函数
class CustomLoss(nn.Module):
    def __init__(self, target_labels, l1_coeff):
        super(CustomLoss, self).__init__()
        self.target_labels = target_labels
        self.l1_coeff = l1_coeff
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        l1_norm = torch.sum(output, dim=1)
        ones = torch.ones_like(l1_norm)
        regularization_term = ((l1_norm - ones)**2).sum()/output.shape[0]
        mse_loss = self.mse_loss(output, target)
        total_loss = regularization_term * self.l1_coeff + mse_loss * 10
        return total_loss

# 定义多层感知机模型
class MFML(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MFML, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        return x


# 定义联合 Auto-sklearn 模型
class JointModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(JointModel, self).__init__()
        self.MFML = MFML(input_size, hidden_size, output_size)
        # 初始化Auto-sklearn回归模型
        self.auto_ml_model = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=3600,  # 任务总时间
            per_run_time_limit=300,  # 每次模型训练的时间限制
            include_estimators=['random_forest', 'k_nearest_neighbors', 'linear_regression',
                                'support_vector_regression', 'decision_tree']
        )
    def fit(self, X_train, y_train):
        self.auto_ml_model.fit(X_train, y_train)

    def forward(self, x):
        outputs = self.auto_ml_model.predict(x)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        MFML_output = self.MFML(outputs_tensor)
        return MFML_output

# 定义需要预测的标签列表
labels_to_predict = ['Clay', 'Cal', 'Dol', 'K_f', 'Na_f', 'Pyrite', 'Quartz']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(feature, lable[labels_to_predict], test_size=0.3, random_state=37)

# 创建研究模型
input_size = len(labels_to_predict)
hidden_size = 1000
output_size = len(labels_to_predict)
model = JointModel(input_size, hidden_size, output_size)

# 使用Auto-sklearn选择最佳模型并拟合数据
for label in labels_to_predict:
    model.fit(X_train, y_train[label])
    # 在测试集上进行预测
    y_test_pred = model.auto_ml_model.predict(X_test)

    # 计算均方误差和 R^2 分数
    test_mse = mean_squared_error(y_test[label], y_test_pred)
    test_r2 = r2_score(y_test[label], y_test_pred)
    test_mae = mean_absolute_error(y_test[label], y_test_pred)

    print(f'Current Label: {label}')
    print(f'Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test R^2: {test_r2:.4f}')
    print('-------------------------------------')

    # 可视化机器学习模型
    explainer = shap.Explainer(model)
    shap_value = explainer(X_train)
    shap.plots.beeswarm(shap_value)

# 定义联合模型
input_size = len(labels_to_predict)
hidden_size = 100
output_size = len(labels_to_predict)
model = JointModel(model, input_size, hidden_size, output_size)

# 定义损失函数和优化器
l1_coeff = 1  # 设置L1正则化系数
criterion = CustomLoss(labels_to_predict, l1_coeff)
# criterion = CustomLoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 转换数据为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    # print(outputs.shape[1])
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_test_pred_tensor = model(X_test_tensor)

sums = y_test_pred_tensor.sum(dim=1)
print(sums)
# 计算模型的性能指标
mse = mean_squared_error(y_test_tensor.numpy(), y_test_pred_tensor.numpy())
r2 = r2_score(y_test_tensor.numpy(), y_test_pred_tensor.numpy())
mae = mean_absolute_error(y_test_tensor.numpy(), y_test_pred_tensor.numpy())

print("\nPerformance of joint LightGBM & MLP Model:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R):", math.sqrt(r2))
print("R-squared (R2):", r2)

