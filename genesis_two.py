import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import pygame
import sys

# ==========================================
# 0. 狩猎场配置 (Configuration) - 修正版
# ==========================================
# 窗口设置
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
GRID_W = WIDTH // GRID_SIZE
GRID_H = HEIGHT // GRID_SIZE
FPS = 60  # 如果想快进，可以把这个改大，比如 300

# 颜色定义
COLOR_BG = (10, 10, 10)        # 黑色背景
COLOR_FOOD = (0, 255, 0)       # 绿色食物
COLOR_AGENT = (200, 50, 50)    # 红色猎人
COLOR_ELITE = (50, 50, 255)    # 蓝色精英
COLOR_GOD = (255, 215, 0)      # 金色半神 (高代际)

# 生存参数 (上帝模式 - 简单难度)
INIT_POPULATION = 30
INIT_ENERGY = 400.0     # [修改] 给予更多初始能量，允许长时间探索
COST_MOVE = 0.1         # [修改] 移动非常便宜
COST_IDLE = 0.1         
COST_GROWTH = 50.0      # 进化大脑消耗
ENERGY_FOOD = 100.0     # [修改] 食物热量很高
MAX_AGE_MUTATION = 50   # 活过多少帧允许突变

# ==========================================
# 1. 猎人大脑 (The Hunter Brain)
# ==========================================
class HunterBrain(nn.Module):
    def __init__(self):
        super(HunterBrain, self).__init__()
        # 输入维度: 2 (食物的相对坐标 dx, dy)
        self.input_dim = 2 
        # 输出维度: 4 (上, 下, 左, 右 的概率)
        self.output_dim = 4
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        self.activation = nn.ReLU() 
        self.out_activation = nn.Softmax(dim=0) 

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return self.out_activation(x)

    def mutate_structure(self):
        # 进化策略：优先加宽，或者加深
        if len(self.layers) == 1:
            self._create_first_hidden_layer()
            return True
        elif self.layers[0].out_features < 32: # 允许大脑长得更大一点
            self._widen_hidden_layer()
            return True
        return False

    def _create_first_hidden_layer(self):
        hidden_size = 8
        layer1 = nn.Linear(self.input_dim, hidden_size)
        layer2 = nn.Linear(hidden_size, self.output_dim)
        # 继承部分权重（虽然结构变了，但尽量保持一点连续性）
        self.layers = nn.ModuleList([layer1, layer2])

    def _widen_hidden_layer(self):
        old_layer1 = self.layers[0]
        old_layer2 = self.layers[1]
        
        in_dim = old_layer1.in_features
        old_hidden = old_layer1.out_features
        new_hidden = old_hidden + 4 
        out_dim = old_layer2.out_features
        
        new_layer1 = nn.Linear(in_dim, new_hidden)
        new_layer2 = nn.Linear(new_hidden, out_dim)
        
        with torch.no_grad():
            new_layer1.weight[:old_hidden, :] = old_layer1.weight
            new_layer1.bias[:old_hidden] = old_layer1.bias
            new_layer2.weight[:, :old_hidden] = old_layer2.weight
            new_layer2.bias[:] = old_layer2.bias
            
        self.layers[0] = new_layer1
        self.layers[1] = new_layer2

# ==========================================
# 2. 猎人个体 (The Agent)
# ==========================================
class Hunter:
    def __init__(self, x, y, generation=0, parent_brain=None):
        self.x = x
        self.y = y
        self.color = COLOR_AGENT
        self.generation = generation
        self.energy = INIT_ENERGY
        self.age = 0
        self.is_alive = True
        
        # 颜色根据代际变化，方便观察
        if self.generation > 0:
            self.color = COLOR_ELITE
        if self.generation > 5:
            self.color = COLOR_GOD
        
        if parent_brain:
            self.brain = copy.deepcopy(parent_brain)
            # 权重突变 (Mutation of Weights)
            with torch.no_grad():
                for param in self.brain.parameters():
                    # 5% 的抖动幅度
                    param.add_(torch.randn_like(param) * 0.05)
        else:
            self.brain = HunterBrain()
            
    def sense_and_act(self, food_list):
        if not self.is_alive: return

        # 1. 感知 (Sensing)
        target_x, target_y = GRID_W // 2, GRID_H // 2 # 默认看向中心
        min_dist = float('inf')
        
        # 寻找最近的食物
        # (为了性能，这里没有做复杂的空间划分，食物太多时可能会稍微慢一点点)
        if food_list:
            # 只是为了演示，简单的遍历
            for fx, fy in food_list:
                dist = (fx - self.x)**2 + (fy - self.y)**2
                if dist < min_dist:
                    min_dist = dist
                    target_x, target_y = fx, fy
        
        # 归一化输入
        dx = (target_x - self.x) / GRID_W
        dy = (target_y - self.y) / GRID_H
        inputs = torch.tensor([dx, dy], dtype=torch.float32)

        # 2. 决策 (Thinking)
        action_probs = self.brain(inputs)
        
        # === [核心修改] 概率采样 ===
        # 使用 Categorical 分布进行采样
        # 这样即使初始概率是 [0.25, 0.25, 0.25, 0.25]，它也会随机乱走，而不是死板不动
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        # 3. 行动 (Moving)
        move_x, move_y = 0, 0
        if action == 0: move_y = -1   # Up
        elif action == 1: move_y = 1  # Down
        elif action == 2: move_x = -1 # Left
        elif action == 3: move_x = 1  # Right
        
        new_x = max(0, min(GRID_W - 1, self.x + move_x))
        new_y = max(0, min(GRID_H - 1, self.y + move_y))
        
        self.x = new_x
        self.y = new_y
        
        # 4. 状态更新
        self.energy -= COST_MOVE
        self.age += 1
        
        # 尝试进化结构 (活得越久越容易进化)
        if self.age > MAX_AGE_MUTATION and self.energy > 200:
            if random.random() < 0.02: # 2% 概率突变结构
                if self.brain.mutate_structure():
                    self.energy -= COST_GROWTH

        if self.energy <= 0:
            self.is_alive = False

# ==========================================
# 3. 游戏主循环
# ==========================================
def run_genesis_two():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genesis-Two: Hunting Grounds (Exploration Mode)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    hunters = []
    for _ in range(INIT_POPULATION):
        hunters.append(Hunter(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)))

    foods = []
    def spawn_food(count):
        for _ in range(count):
            foods.append((random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)))
    
    # [修改] 初始投放 200 个食物，满屏都是
    spawn_food(200) 

    running = True
    ticks = 0
    max_gen = 0
    
    while running:
        screen.fill(COLOR_BG)
        ticks += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 逻辑更新 ---
        active_hunters = [h for h in hunters if h.is_alive]
        
        # 食物集合用于快速查找 (优化性能)
        food_set = set(foods)
        
        for hunter in active_hunters:
            hunter.sense_and_act(foods) # 传入列表用于感知
            
            # 碰撞检测 (吃食物)
            if (hunter.x, hunter.y) in food_set:
                food_set.remove((hunter.x, hunter.y))
                hunter.energy += ENERGY_FOOD
                
                # 繁殖逻辑
                if hunter.energy > 300: # 能量够了就生
                    hunter.energy -= 150
                    # 孩子出生在附近
                    cx = max(0, min(GRID_W-1, hunter.x + random.choice([-1,0,1])))
                    cy = max(0, min(GRID_H-1, hunter.y + random.choice([-1,0,1])))
                    
                    child = Hunter(cx, cy, generation=hunter.generation+1, parent_brain=hunter.brain)
                    hunters.append(child)
                    if child.generation > max_gen:
                        max_gen = child.generation
        
        # 同步回列表
        foods = list(food_set)
        
        # 自动补充食物 (保持生态平衡)
        if len(foods) < 200:
            spawn_food(200 - len(foods))

        # 清理尸体
        hunters = [h for h in hunters if h.is_alive]
        
        # 灭绝保护
        if len(hunters) < 5:
            hunters.append(Hunter(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)))

        # --- 渲染 ---
        for fx, fy in foods:
            pygame.draw.rect(screen, COLOR_FOOD, (fx * GRID_SIZE, fy * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        for h in hunters:
            pygame.draw.rect(screen, h.color, (h.x * GRID_SIZE, h.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # UI
        status_text = f"Pop: {len(hunters)} | Max Gen: {max_gen} | Time: {ticks}"
        screen.blit(font.render(status_text, True, (255, 255, 255)), (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_genesis_two()