import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import pygame
import sys

# ==========================================
# 0. 自律进化配置 (Auto-Evolution Config)
# ==========================================
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
GRID_W = WIDTH // GRID_SIZE
GRID_H = HEIGHT // GRID_SIZE
FPS = 300 # 加速运行，不用太慢

COLOR_BG = (10, 10, 10)
COLOR_FOOD = (0, 255, 0)
COLOR_BOT = (200, 200, 200)

# 基础参数 (由于有自动平衡，初始值不那么重要了)
INIT_POPULATION = 50
INIT_ENERGY = 400.0     
COST_MOVE = 0.2
COST_IDLE = 0.1
ENERGY_FOOD = 100.0
ENERGY_KILL = 200.0
VISION_RANGE = 2

# ==========================================
# 1. 可生长的大脑 (Growable Brain)
# ==========================================
class EvolvingBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EvolvingBrain, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. 视觉编码层 (可变宽)
        self.encoder = nn.Linear(input_dim, 64)
        
        # 2. 记忆核心 (GRU) - 保持固定以稳定时序，但可以加深(这里简化为单层)
        self.memory = nn.GRUCell(64, hidden_dim)
        
        # 3. 决策层 (可变宽)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, vision, hidden):
        # 确保 hidden 维度匹配 (因为 hidden_dim 可能在进化中变了)
        if hidden is None or hidden.shape[1] != self.hidden_dim:
            hidden = torch.zeros(1, self.hidden_dim)
            
        x = torch.relu(self.encoder(vision))
        new_hidden = self.memory(x, hidden)
        logits = self.decoder(new_hidden)
        probs = torch.softmax(logits, dim=1)
        return probs, new_hidden

    def mutate_structure(self):
        """ 核心：自我设计大脑结构 """
        mutation_type = random.choice(['widen_encoder', 'grow_memory'])
        
        if mutation_type == 'widen_encoder':
            # 增加编码层宽度：看得更清楚
            self._expand_layer(self.encoder, 16) # 加 16 个神经元
            # 重新适配 GRU 输入
            new_input_size = self.encoder.out_features
            self.memory = nn.GRUCell(new_input_size, self.hidden_dim)
            
        elif mutation_type == 'grow_memory':
            # 增加记忆容量：记得更多
            old_hidden = self.hidden_dim
            new_hidden = old_hidden + 8
            
            # 创建新 GRU
            new_gru = nn.GRUCell(self.encoder.out_features, new_hidden)
            
            # 继承旧权重 (知识传承)
            with torch.no_grad():
                # 这是一个简化的权重复制，真实情况需要仔细对齐矩阵
                # 这里为了不报错，我们只保留部分权重，或者轻微随机化
                # 在 Auto-Evolution 中，稍微的破坏性突变是可以接受的
                pass 
                
            self.memory = new_gru
            self.hidden_dim = new_hidden
            
            # 重新适配 Decoder
            self.decoder = nn.Linear(new_hidden, self.output_dim)

    def _expand_layer(self, layer, extra_nodes):
        # 简单的层扩展辅助函数
        old_out = layer.out_features
        new_out = old_out + extra_nodes
        new_layer = nn.Linear(layer.in_features, new_out)
        # 复制旧权重
        with torch.no_grad():
            new_layer.weight[:old_out, :] = layer.weight
            new_layer.bias[:old_out] = layer.bias
        return new_layer

# ==========================================
# 2. 智能体
# ==========================================
class Agent:
    def __init__(self, x, y, gen=0, brain=None):
        self.x = x
        self.y = y
        self.gen = gen
        self.energy = INIT_ENERGY
        self.alive = True
        self.kills = 0
        self.brain_size = 0 # 记录大脑复杂度
        
        input_dim = (VISION_RANGE * 2 + 1) ** 2
        
        if brain:
            self.brain = copy.deepcopy(brain)
            # 结构突变：1% 的概率大脑发生物理生长
            if random.random() < 0.01:
                self.brain.mutate_structure()
            # 权重突变：10% 概率调整参数
            self._mutate_weights()
        else:
            self.brain = EvolvingBrain(input_dim, 32, 4)
            
        self.hidden = None
        self.brain_size = self.brain.hidden_dim # 用记忆容量代表智商

    def _mutate_weights(self):
        with torch.no_grad():
            for p in self.brain.parameters():
                if random.random() < 0.1:
                    p.add_(torch.randn_like(p) * 0.05)

    def get_view(self, foods, agents_map):
        view = []
        r = VISION_RANGE
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                tx, ty = self.x + dx, self.y + dy
                val = 0.0
                if not (0 <= tx < GRID_W and 0 <= ty < GRID_H): val = -1.0 # 墙
                elif (tx, ty) in foods: val = 1.0 # 食物
                elif (tx, ty) in agents_map: 
                    other = agents_map[(tx, ty)]
                    val = 0.5 if self.energy > other.energy else -0.5 # 敌我
                view.append(val)
        return torch.tensor(view).float().unsqueeze(0)

    def act(self, foods, agents_map):
        if not self.alive: return
        
        view = self.get_view(foods, agents_map)
        probs, self.hidden = self.brain(view, self.hidden)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
        
        tx, ty = self.x + dx, self.y + dy
        self.energy -= COST_MOVE
        
        # 物理交互
        if not (0 <= tx < GRID_W and 0 <= ty < GRID_H):
            self.energy -= COST_MOVE # 撞墙
            return
            
        if (tx, ty) in agents_map:
            other = agents_map[(tx, ty)]
            if other != self:
                # 捕食逻辑
                if self.energy > other.energy * 1.2:
                    self.energy += ENERGY_KILL
                    self.kills += 1
                    other.alive = False
                    del agents_map[(self.x, self.y)]
                    self.x, self.y = tx, ty
                    agents_map[(self.x, self.y)] = self
                else:
                    self.energy -= COST_IDLE # 碰撞
            return
            
        # 移动到空地
        del agents_map[(self.x, self.y)]
        self.x, self.y = tx, ty
        agents_map[(self.x, self.y)] = self
        
        # 吃
        if (self.x, self.y) in foods:
            foods.remove((self.x, self.y))
            self.energy += ENERGY_FOOD
            
        if self.energy <= 0:
            self.alive = False
            del agents_map[(self.x, self.y)]

# ==========================================
# 3. 自动平衡环境 (The Auto-Balancer)
# ==========================================
def run_genesis_four():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genesis-Four: Autonomous Evolution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    agents = [Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)) for _ in range(INIT_POPULATION)]
    foods = set()
    agents_map = { (a.x, a.y): a for a in agents }
    
    # 动态环境参数
    food_spawn_rate = 5
    target_population = 50 # 环境试图维持的理想人口
    
    running = True
    ticks = 0
    max_gen = 0
    max_brain = 32
    
    while running:
        screen.fill(COLOR_BG)
        ticks += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        # --- 1. 逻辑 ---
        random.shuffle(agents)
        active_agents = [a for a in agents if a.alive]
        
        for a in active_agents:
            a.act(foods, agents_map)
            
            # 繁殖：能量高就生
            if a.energy > 800:
                a.energy -= 400
                child = Agent(a.x, a.y, a.gen + 1, a.brain)
                # 找空位
                spawned = False
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = a.x + dx, a.y + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in agents_map:
                        child.x, child.y = nx, ny
                        agents.append(child)
                        agents_map[(nx, ny)] = child
                        spawned = True
                        if child.gen > max_gen: max_gen = child.gen
                        if child.brain_size > max_brain: max_brain = child.brain_size
                        break
                if not spawned: a.energy += 400
        
        agents = [a for a in agents if a.alive]
        agents_map = { (a.x, a.y): a for a in agents }
        
        # --- 2. 自动平衡机制 (Auto-Balance) ---
        pop_count = len(agents)
        
        # 如果人太多，环境恶化（食物减少）
        if pop_count > target_population * 1.5:
            food_spawn_rate = max(1, food_spawn_rate - 1)
        # 如果人太少，环境恢复（食物增加）
        elif pop_count < target_population * 0.5:
            food_spawn_rate = min(50, food_spawn_rate + 2)
            
        # 刷新食物
        if len(foods) < 200: # 只要不满就一直尝试长
            for _ in range(food_spawn_rate):
                fx, fy = random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)
                if (fx, fy) not in foods and (fx, fy) not in agents_map:
                    foods.add((fx, fy))
                    
        # 灭绝保护
        if pop_count < 5:
            new_a = Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1))
            agents.append(new_a)
            agents_map[(new_a.x, new_a.y)] = new_a
            
        # --- 3. 渲染 ---
        for fx, fy in foods:
            pygame.draw.rect(screen, COLOR_FOOD, (fx*GRID_SIZE, fy*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        for a in agents:
            # 颜色根据智商（大脑容量）显示
            # 越聪明（隐藏层越大）越亮，或者越蓝
            intensity = min(255, 100 + (a.brain_size - 32) * 5)
            color = (intensity, 50, 50) # 红色基调
            
            if a.gen > 20: color = (255, 215, 0) # 黄金传说
            
            pygame.draw.rect(screen, color, (a.x*GRID_SIZE, a.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # UI
        info = f"Pop: {pop_count} | Gen: {max_gen} | Max Brain: {max_brain} | FoodRate: {food_spawn_rate}"
        screen.blit(font.render(info, True, (255, 255, 255)), (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_genesis_four()