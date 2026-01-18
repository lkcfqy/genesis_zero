import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. 基础配置与环境 (The Environment)
# ==========================================

# 设定随机种子，保证每次运行结果可复现
torch.manual_seed(42)
np.random.seed(42)

# XOR 数据集：经典的非线性问题
# 输入: [0,0], [0,1], [1,0], [1,1]
# 输出: [0],   [1],   [1],   [0]
X_RAW = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y_RAW = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print(">>> 环境初始化完成: 任务目标 XOR (异或)")

# ==========================================
# 2. 动态神经网络 (The Self-Modifying Agent)
# ==========================================

class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        # 初始状态：极简网络
        # 输入层(2) -> 输出层(1)
        # 这种线性结构从数学上绝对无法解决 XOR 问题
        self.input_dim = 2
        self.hidden_dim = 0  # 初始没有隐藏层
        self.output_dim = 1
        
        # 定义层
        # 我们使用 Parameter 手动定义权重，方便进行"手术"
        self.layers = nn.ModuleList()
        # 初始只有一层: Input -> Output
        self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        
        self.activation = nn.ReLU()

    def forward(self, x):
        # 前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 如果不是最后一层，加激活函数
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.sigmoid(x) # 输出 0-1 之间的概率

    def evolve_structure(self):
        """
        核心进化逻辑：自我修改结构
        这里模拟'架构师'的操作：在网络中间插入/扩展隐藏层
        """
        print(f"\n[进化突变] 正在修改大脑结构...")
        
        # 策略：如果还没有隐藏层，创建一个；如果有，加宽它。
        if self.hidden_dim == 0:
            self._create_first_hidden_layer()
        else:
            self._add_neuron_to_hidden()

    def _create_first_hidden_layer(self):
        """从 0 隐藏层变为 2 个神经元的隐藏层"""
        # 为什么是 2？解决 XOR 最少需要 2 个隐藏节点
        new_hidden_size = 2
        print(f" -> 动作: 插入新的隐藏层 (Size: {new_hidden_size})")
        
        # 创建新层
        layer1 = nn.Linear(self.input_dim, new_hidden_size)
        layer2 = nn.Linear(new_hidden_size, self.output_dim)
        
        # 替换旧的 ModuleList
        self.layers = nn.ModuleList([layer1, layer2])
        self.hidden_dim = new_hidden_size

    def _add_neuron_to_hidden(self):
        """给现有的隐藏层增加神经元 (保留旧知识)"""
        current_hidden = self.hidden_dim
        new_hidden = current_hidden + 1
        print(f" -> 动作: 增加神经元 {current_hidden} -> {new_hidden}")
        
        # 获取旧层
        old_layer1 = self.layers[0]
        old_layer2 = self.layers[1]
        
        # --- 手术开始 ---
        # 1. 创建新层
        new_layer1 = nn.Linear(self.input_dim, new_hidden)
        new_layer2 = nn.Linear(new_hidden, self.output_dim)
        
        # 2. 知识传承 (Copy Weights)
        # 将旧权重复指给新层，保证并没有"失忆"
        with torch.no_grad():
            # Layer 1: 复制旧的行，新加的一行随机初始化
            new_layer1.weight[:current_hidden, :] = old_layer1.weight
            new_layer1.bias[:current_hidden] = old_layer1.bias
            
            # Layer 2: 复制旧的列，新加的一列随机初始化
            new_layer2.weight[:, :current_hidden] = old_layer2.weight
            # bias 不受输入维度影响，直接复制
            new_layer2.bias[:] = old_layer2.bias
            
        # 3. 替换身体部件
        self.layers = nn.ModuleList([new_layer1, new_layer2])
        self.hidden_dim = new_hidden

# ==========================================
# 3. 进化引擎 (The Evolutionary Engine)
# ==========================================

def train_and_evolve():
    # 初始化智能体
    agent = DynamicNet()
    criterion = nn.MSELoss()
    
    # 进化参数
    max_epochs = 20000
    patience = 2000      # 如果训练这么久 loss 还不降，就触发进化
    stagnation_counter = 0
    best_loss = 1.0
    
    optimizer = optim.Adam(agent.parameters(), lr=0.01)

    loss_history = []

    print(f"Start Structure: Input({agent.input_dim}) -> Output({agent.output_dim})")

    for epoch in range(max_epochs):
        # --- 标准训练步骤 ---
        optimizer.zero_grad()
        outputs = agent(X_RAW)
        loss = criterion(outputs, Y_RAW)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        loss_history.append(curr_loss)

        # --- 监控与进化判断 ---
        if curr_loss < best_loss - 0.001:
            best_loss = curr_loss
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # 打印进度
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {curr_loss:.4f} | Hidden Nodes = {agent.hidden_dim}")

        # 成功条件 (XOR 解决)
        if curr_loss < 0.02:
            print(f"\n>>> 成功! 智能涌现。在 Epoch {epoch} 解决问题。")
            print(f"最终预测:\n{outputs.detach().numpy().round(2)}")
            break

        # --- 触发进化 (The Architect kicks in) ---
        if stagnation_counter >= patience:
            print(f"\n[警告] 智力陷入瓶颈 (Loss停滞在 {curr_loss:.4f})")
            
            # 这里的逻辑是：如果现在的脑子学不会，那就换个脑子
            agent.evolve_structure()
            
            # 进化后需要重新初始化优化器，因为参数变了
            optimizer = optim.Adam(agent.parameters(), lr=0.01)
            stagnation_counter = 0
            # 稍微重置 best_loss 以避免立即再次触发
            best_loss = curr_loss 

    return loss_history, agent

# ==========================================
# 4. 运行与可视化
# ==========================================

if __name__ == "__main__":
    history, final_agent = train_and_evolve()
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("Evolution of Intelligence: Loss vs Time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Error)")
    
    # 标记进化点 (简单示意，Loss突然波动的地方通常是进化点)
    plt.grid(True)
    plt.savefig("evolution_chart.png")
    print("\n>>> 图表已保存为 evolution_chart.png")
    print(f"最终网络结构: Input(2) -> Hidden({final_agent.hidden_dim}) -> Output(1)")