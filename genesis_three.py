import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import pygame
import sys

# ==========================================
# 0. 黑暗森林配置 (天堂版 - Easy Mode)
# ==========================================
# 窗口设置
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
GRID_W = WIDTH // GRID_SIZE
GRID_H = HEIGHT // GRID_SIZE
FPS = 60 # 如果想加速进化，可以把这个改大，比如 300

# 颜色定义
COLOR_BG = (20, 20, 20)        # 深灰背景
COLOR_WALL = (100, 100, 100)   # 墙壁
COLOR_FOOD = (0, 255, 0)       # 绿色食物
COLOR_WEAK = (150, 150, 150)   # 弱者(灰色)
COLOR_STRONG = (255, 50, 50)   # 强者(红色)
COLOR_LEGEND = (255, 215, 0)   # 传说级(金色)

# 生存参数 (调整为易于存活，促进早期进化)
INIT_POPULATION = 60    # [修改] 初始人口增多
INIT_ENERGY = 600.0     # [修改] 超高初始能量 (给笨蛋更多尝试时间)
COST_MOVE = 0.1         # [修改] 移动几乎免费
COST_IDLE = 0.1         
COST_ATTACK = 2.0       # [修改] 攻击成本降低
ENERGY_FOOD = 150.0     # [修改] 一个食物管很久
ENERGY_KILL = 200.0     # 猎杀奖励
VISION_RANGE = 2        # 视野半径 (2格 => 5x5视野)

# ==========================================
# 1. 记忆大脑 (The Memory Brain - GRU)
# ==========================================
class RecurrentBrain(nn.Module):
    def __init__(self):
        super(RecurrentBrain, self).__init__()
        
        # 输入: 5x5 视野 = 25 个像素点
        # 每个点的值: 0(空), 1(食), -1(墙), 0.5(弱者), -0.5(强者)
        self.input_dim = (VISION_RANGE * 2 + 1) ** 2 
        self.hidden_dim = 32 # 记忆容量
        self.output_dim = 4  # 上下左右
        
        # 视觉编码
        self.encoder = nn.Linear(self.input_dim, 64)
        
        # 记忆核心 (GRU Cell)
        self.memory_cell = nn.GRUCell(64, self.hidden_dim)
        
        # 决策层
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, vision, hidden_state):
        # 1. 编码视觉
        x = torch.relu(self.encoder(vision))
        
        # 2. 更新记忆
        new_hidden = self.memory_cell(x, hidden_state)
        
        # 3. 决策
        action_logits = self.decoder(new_hidden)
        action_probs = self.softmax(action_logits)
        
        return action_probs, new_hidden

    def mutate(self):
        # 权重突变
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < 0.05: # 5% 概率突变某个权重张量
                    param.add_(torch.randn_like(param) * 0.1)

# ==========================================
# 2. 智能体 (The Agent)
# ==========================================
class Agent:
    def __init__(self, x, y, generation=0, brain=None):
        self.x = x
        self.y = y
        self.generation = generation
        self.energy = INIT_ENERGY
        self.age = 0
        self.is_alive = True
        self.kills = 0
        
        # 大脑与记忆
        if brain:
            self.brain = copy.deepcopy(brain)
            self.brain.mutate() 
        else:
            self.brain = RecurrentBrain()
            
        # 初始化短期记忆 (全0状态)
        self.hidden_state = torch.zeros(1, self.brain.hidden_dim)

    def get_local_view(self, food_set, agent_map):
        view = []
        r = VISION_RANGE
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                tx, ty = self.x + dx, self.y + dy
                
                val = 0.0
                # 1. 边界/墙
                if tx < 0 or tx >= GRID_W or ty < 0 or ty >= GRID_H:
                    val = -1.0
                # 2. 食物
                elif (tx, ty) in food_set:
                    val = 1.0
                # 3. 其他生物
                elif (tx, ty) in agent_map:
                    other = agent_map[(tx, ty)]
                    if other != self:
                        # 简单的敌我识别信号
                        if self.energy > other.energy:
                            val = 0.5  # 猎物
                        else:
                            val = -0.5 # 威胁
                view.append(val)
        return torch.tensor(view, dtype=torch.float32).unsqueeze(0) 

    def act(self, food_set, agent_map):
        if not self.is_alive: return

        # 1. 感知
        vision = self.get_local_view(food_set, agent_map)
        
        # 2. 思考
        probs, self.hidden_state = self.brain(vision, self.hidden_state)
        
        # 3. 决策 (概率采样)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
        # 4. 执行
        dx, dy = 0, 0
        if action == 0: dy = -1 # Up
        elif action == 1: dy = 1  # Down
        elif action == 2: dx = -1 # Left
        elif action == 3: dx = 1  # Right
        
        target_x, target_y = self.x + dx, self.y + dy
        
        self.energy -= COST_MOVE
        self.age += 1
        
        # --- 物理引擎 ---
        
        # A. 撞墙
        if target_x < 0 or target_x >= GRID_W or target_y < 0 or target_y >= GRID_H:
            self.energy -= COST_MOVE # 撞墙轻微惩罚
            return 

        # B. 撞人 (博弈)
        if (target_x, target_y) in agent_map:
            other = agent_map[(target_x, target_y)]
            if other != self:
                # 判定捕食: 能量必须显著高于对方
                if self.energy > other.energy * 1.2:
                    # 吃掉对方
                    self.energy += ENERGY_KILL
                    self.energy -= COST_ATTACK
                    self.kills += 1
                    other.is_alive = False 
                    # 移动到对方位置
                    del agent_map[(self.x, self.y)]
                    self.x, self.y = target_x, target_y
                    agent_map[(self.x, self.y)] = self
                else:
                    # 势均力敌或更弱，发生碰撞
                    self.energy -= COST_ATTACK
                    other.energy -= COST_ATTACK
            return

        # C. 移动到空地
        del agent_map[(self.x, self.y)]
        self.x, self.y = target_x, target_y
        agent_map[(self.x, self.y)] = self
        
        # D. 吃食物
        if (self.x, self.y) in food_set:
            food_set.remove((self.x, self.y))
            self.energy += ENERGY_FOOD

        # 死亡判定
        if self.energy <= 0:
            self.is_alive = False
            if (self.x, self.y) in agent_map:
                del agent_map[(self.x, self.y)]

# ==========================================
# 3. 游戏主循环
# ==========================================
def run_dark_forest():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genesis-Three: Paradise Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    agents = [Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)) for _ in range(INIT_POPULATION)]
    foods = set()
    
    agent_map = {} 
    for a in agents:
        agent_map[(a.x, a.y)] = a

    def spawn_food(count):
        attempts = 0
        while count > 0 and attempts < 2000:
            fx, fy = random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)
            if (fx, fy) not in foods and (fx, fy) not in agent_map:
                foods.add((fx, fy))
                count -= 1
            attempts += 1
            
    # [修改] 初始食物铺满，不再稀缺
    spawn_food(200) 

    running = True
    ticks = 0
    max_gen = 0
    top_kills = 0
    
    while running:
        screen.fill(COLOR_BG)
        ticks += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 逻辑更新 ---
        random.shuffle(agents)
        active_agents = [a for a in agents if a.is_alive]
        
        for agent in active_agents:
            agent.act(foods, agent_map)
            
            # 繁殖逻辑
            if agent.energy > 800: # 能量积累到一定程度
                agent.energy -= 400
                child = Agent(agent.x, agent.y, agent.generation + 1, agent.brain)
                
                # 在周围找空位生孩子
                spawned = False
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = agent.x + dx, agent.y + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in agent_map:
                        child.x, child.y = nx, ny
                        agents.append(child)
                        agent_map[(nx, ny)] = child
                        spawned = True
                        if child.generation > max_gen: max_gen = child.generation
                        break
                if not spawned:
                    agent.energy += 400 

        agents = [a for a in agents if a.is_alive]
        agent_map = { (a.x, a.y): a for a in agents }
        
        if agents:
            top_kills = max(a.kills for a in agents)
        
        # [核心修改] 保持食物极其充足 (>150)
        # 确保只要它们肯动，大概率能撞到吃的
        if len(foods) < 150:
            spawn_food(10)

        # 灭绝保护
        if len(agents) < 10:
            new_a = Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1))
            agents.append(new_a)
            agent_map[(new_a.x, new_a.y)] = new_a

        # --- 渲染 ---
        for fx, fy in foods:
            pygame.draw.rect(screen, COLOR_FOOD, (fx*GRID_SIZE, fy*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        for a in agents:
            color = COLOR_WEAK
            if a.energy > 500: color = COLOR_STRONG
            if a.generation > 10: color = COLOR_LEGEND
            
            rect = (a.x*GRID_SIZE, a.y*GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            if a.kills > 0: # 杀手带红框
                pygame.draw.rect(screen, (255, 0, 0), rect, 2)

        info = f"Pop: {len(agents)} | Gen: {max_gen} | Top Kills: {top_kills} | Food: {len(foods)}"
        screen.blit(font.render(info, True, (255, 255, 255)), (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_dark_forest()