import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt

# ==========================================
# 0. 世界法则 (Global Configuration)
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# XOR 数据集
X_RAW = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y_RAW = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 生存参数
INIT_POPULATION = 20
MAX_POPULATION = 50
INIT_ENERGY = 200.0     # [修改] 给更多初始能量 (原来是 50)
COST_PER_TICK = 0.5     # [修改] 降低代谢消耗 (原来是 1.0)
COST_GROWTH = 20.0      # [修改] 长脑子代价高一点，防止乱长
REWARD_SOLVED = 200.0   # 解决问题的奖励
REWARD_ACCURACY = 5.0   # 答对一部分的奖励

# ==========================================
# 1. 核心大脑 (The Brain - Same as Genesis-Zero)
# ==========================================
class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.input_dim = 2
        self.hidden_dim = 0 
        self.output_dim = 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.sigmoid(x)

    def mutate_structure(self):
        # 简化版突变：优先加层，有了层优先加节点
        if self.hidden_dim == 0:
            self._create_first_hidden_layer()
            return True # 消耗能量标志
        elif self.hidden_dim < 5: # 限制一下最大脑容量防止内存爆炸
            self._add_neuron_to_hidden()
            return True
        return False

    def _create_first_hidden_layer(self):
        new_hidden_size = 2
        layer1 = nn.Linear(self.input_dim, new_hidden_size)
        layer2 = nn.Linear(new_hidden_size, self.output_dim)
        self.layers = nn.ModuleList([layer1, layer2])
        self.hidden_dim = new_hidden_size

    def _add_neuron_to_hidden(self):
        current_hidden = self.hidden_dim
        new_hidden = current_hidden + 1
        old_layer1 = self.layers[0]
        old_layer2 = self.layers[1]
        
        new_layer1 = nn.Linear(self.input_dim, new_hidden)
        new_layer2 = nn.Linear(new_hidden, self.output_dim)
        
        with torch.no_grad():
            new_layer1.weight[:current_hidden, :] = old_layer1.weight
            new_layer1.bias[:current_hidden] = old_layer1.bias
            new_layer2.weight[:, :current_hidden] = old_layer2.weight
            new_layer2.bias[:] = old_layer2.bias
            
        self.layers = nn.ModuleList([new_layer1, new_layer2])
        self.hidden_dim = new_hidden

# ==========================================
# 2. 生物体 (The Organism)
# ==========================================
class Organism:
    def __init__(self, generation=0, parent_brain=None):
        self.generation = generation
        self.age = 0
        self.energy = INIT_ENERGY
        self.is_alive = True
        self.solved = False
        self.best_loss = 1.0
        
        # 获得大脑
        if parent_brain:
            # 文化传承：完全克隆父母的大脑
            self.brain = copy.deepcopy(parent_brain)
            # 哪怕是克隆的，也要稍微变异一点点权重（个性）
            with torch.no_grad():
                for param in self.brain.parameters():
                    param.add_(torch.randn_like(param) * 0.05)
        else:
            # 原始人：白板大脑
            self.brain = DynamicNet()
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def live_one_tick(self):
        if not self.is_alive: return
        
        self.age += 1
        self.energy -= COST_PER_TICK
        
        # 1. 学习 (Thinking)
        self.optimizer.zero_grad()
        output = self.brain(X_RAW)
        loss = self.criterion(output, Y_RAW)
        loss.backward()
        self.optimizer.step()
        
        curr_loss = loss.item()
        
        # 2. 获得奖励 (Feeding)
        # 如果 Loss 很低，说明"捕猎"成功，获得能量
        if curr_loss < 0.24:
            self.energy += REWARD_ACCURACY
        
        # 记录最佳状态
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            
        # 3. 进化决策 (Self-Architecture)
        # 如果卡住了（Loss 不降）且能量充足，尝试长脑子
        if curr_loss > 0.05 and self.age > 20 and self.energy > 50:
            # 只有 5% 的概率会突然想改变结构（避免所有人都同时突变）
            if random.random() < 0.02:
                did_grow = self.brain.mutate_structure()
                if did_grow:
                    self.energy -= COST_GROWTH # 长脑子消耗大量能量
                    # 重新初始化优化器因为参数变了
                    self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)

        # 4. 判定是否"悟道" (Solved)
        if curr_loss < 0.02:
            self.solved = True
            self.energy += REWARD_SOLVED # 巨大的生存奖励

        # 5. 死亡判定
        if self.energy <= 0:
            self.is_alive = False

# ==========================================
# 3. 创世纪引擎 (The World Engine)
# ==========================================
class GenesisWorld:
    def __init__(self):
        self.population = []
        self.epoch = 0
        self.history_pop = []
        self.history_avg_loss = []
        
        # 亚当与夏娃：初始化种群
        print(f"🌍 创世纪启动... 投放 {INIT_POPULATION} 个原始生物")
        for _ in range(INIT_POPULATION):
            self.population.append(Organism(generation=0))

    def update(self):
        self.epoch += 1
        
        # 1. 所有生物行动一轮
        alive_count = 0
        total_loss = 0
        solvers = 0
        
        for org in self.population:
            if org.is_alive:
                org.live_one_tick()
                if org.is_alive: # 行动后可能累死了
                    alive_count += 1
                    total_loss += org.best_loss
                    if org.solved:
                        solvers += 1

        # 2. 清理尸体
        self.population = [org for org in self.population if org.is_alive]
        
        # 3. 繁衍 (Reproduction) - 只有最强壮的才能生孩子
        # 筛选条件：解决了问题，或者能量很高
        elites = [org for org in self.population if org.solved or org.energy > 80]
        
        new_babies = []
        # 如果人口不足且有精英，开始繁殖
        if len(self.population) < MAX_POPULATION and len(elites) > 0:
            for parent in elites:
                # 消耗父母能量生孩子
                if parent.energy > 60: 
                    parent.energy -= 30
                    # 孩子继承父母的 generation + 1，以及父母的大脑
                    child = Organism(generation=parent.generation + 1, parent_brain=parent.brain)
                    new_babies.append(child)
        
        self.population.extend(new_babies)

        # 4. 灭绝保护 (如果人都死光了，投放新的原始人)
        if len(self.population) < 5:
            print("⚠️ 种群濒临灭绝! 投放新的原始人...")
            for _ in range(5):
                self.population.append(Organism(generation=0))

        # 5. 数据记录
        avg_loss = total_loss / alive_count if alive_count > 0 else 1.0
        self.history_pop.append(len(self.population))
        self.history_avg_loss.append(avg_loss)

        return alive_count, avg_loss, solvers, len(new_babies)

# ==========================================
# 4. 运行模拟
# ==========================================
if __name__ == "__main__":
    world = GenesisWorld()
    
    try:
        start_time = time.time()
        for i in range(1000): # 运行 1000 个世界时刻
            pop_count, loss, solvers, babies = world.update()
            
            # 这是一个简单的控制台可视化
            # 打印频率不要太高
            if i % 10 == 0:
                # 找出当前最高代际
                max_gen = max([p.generation for p in world.population]) if world.population else 0
                print(f"Tick {i:4d} | Pop: {pop_count:2d} (Babies: {babies}) | Avg Loss: {loss:.4f} | Solvers: {solvers} | Max Gen: {max_gen}")
            
            # 如果大部分人都解决了问题，提前结束
            if solvers > 10:
                print(f"\n🚀 文明等级突破! 超过 10 个个体已觉醒。模拟在 Tick {i} 停止。")
                break
                
    except KeyboardInterrupt:
        print("\n模拟手动停止。")

    # 绘图分析
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (Ticks)')
    ax1.set_ylabel('Population', color=color)
    ax1.plot(world.history_pop, color=color, alpha=0.6, label='Population')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Average Loss (Intelligence)', color=color)
    ax2.plot(world.history_avg_loss, color=color, linewidth=2, label='Avg Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    # 反转 Loss 轴，向上代表更聪明
    ax2.set_ylim(0, 0.5)
    ax2.invert_yaxis() 

    plt.title("Genesis-One: Evolution of a Neural Society")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("genesis_one_chart.png")
    print(">>> 进化历史图表已保存为 genesis_one_chart.png")