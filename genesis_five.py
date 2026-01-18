import torch
import torch.nn as nn
import numpy as np
import copy
import random
import pygame
import sys

# ==========================================
# 0. 部落冲突配置
# ==========================================
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
GRID_W = WIDTH // GRID_SIZE
GRID_H = HEIGHT // GRID_SIZE
FPS = 120

COLOR_BG = (10, 10, 10)
COLOR_FOOD = (0, 255, 0)      
COLOR_MAMMOTH = (148, 0, 211) 

INIT_POPULATION = 80    # 人多一点，冲突更明显
INIT_ENERGY = 600.0
COST_MOVE = 0.1
COST_SIGNAL = 0.01      
ENERGY_FOOD = 20.0      # 草根本吃不饱
ENERGY_MAMMOTH = 1200.0 # 必须吃肉
VISION_RANGE = 2

# ==========================================
# 1. 社交大脑
# ==========================================
class SocialBrain(nn.Module):
    def __init__(self):
        super(SocialBrain, self).__init__()
        # 视觉(25) + 听觉(1)
        self.input_dim = (VISION_RANGE * 2 + 1) ** 2 + 1
        self.hidden_dim = 32
        self.output_dim = 5 # 上下左右 + 说话
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, vision, hearing):
        x = torch.cat([vision, hearing], dim=1)
        logits = self.net(x)
        action_probs = torch.softmax(logits[:, :4], dim=1)
        signal_val = torch.sigmoid(logits[:, 4:])
        return action_probs, signal_val

    def mutate(self):
        with torch.no_grad():
            for p in self.net.parameters():
                if random.random() < 0.1:
                    p.add_(torch.randn_like(p) * 0.1)

# ==========================================
# 2. 智能体 (强制部落版)
# ==========================================
class Agent:
    def __init__(self, x, y, gen=0, brain=None, parent_signal=None):
        self.x = x
        self.y = y
        self.gen = gen
        self.energy = INIT_ENERGY
        self.alive = True
        
        # [修改] 强制初始化为极端颜色
        if parent_signal is not None:
            # 遗传：稍微变异一点点，但大致保持颜色
            self.signal = parent_signal + random.uniform(-0.05, 0.05)
            self.signal = max(0.0, min(1.0, self.signal))
        else:
            # 祖先：随机选边站
            self.signal = 1.0 if random.random() < 0.5 else 0.0
        
        if brain:
            self.brain = copy.deepcopy(brain)
            self.brain.mutate()
        else:
            self.brain = SocialBrain()

    def get_view(self, foods, mammoths, agents_map):
        view = []
        r = VISION_RANGE
        nearest_signal = 0.0 # 默认
        min_dist = 999
        
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                tx, ty = self.x + dx, self.y + dy
                val = 0.0
                
                if not (0 <= tx < GRID_W and 0 <= ty < GRID_H): val = -1.0 # 墙
                elif (tx, ty) in foods: val = 0.5 # 草
                elif (tx, ty) in mammoths: val = 1.0 # 肉
                elif (tx, ty) in agents_map:
                    other = agents_map[(tx, ty)]
                    if other != self:
                        # [修改] 视觉里也能看到对方是红是蓝
                        # 红色队友显示为 -0.8，蓝色队友显示为 -0.2
                        val = -0.8 if other.signal > 0.5 else -0.2
                        
                        dist = abs(dx) + abs(dy)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_signal = other.signal
                
                view.append(val)
                
        return torch.tensor(view).float().unsqueeze(0), torch.tensor([[nearest_signal]]).float()

    def act(self, foods, mammoths, agents_map):
        if not self.alive: return
        
        vision, hearing = self.get_view(foods, mammoths, agents_map)
        probs, signal_out = self.brain(vision, hearing)
        
        # 说话 (但这不改变自己的固有肤色，只改变发出的信号)
        # 为了演示，我们让肤色=固有信号，而发出的声音影响由于时间关系简化处理
        # 这里我们让 signal 慢慢趋向于神经网络的输出，实现"变色龙"效果
        # 但为了保留部落特征，我们让改变很慢
        self.signal = self.signal * 0.9 + signal_out.item() * 0.1
        
        self.energy -= COST_SIGNAL
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
        
        tx, ty = self.x + dx, self.y + dy
        self.energy -= COST_MOVE
        
        if not (0 <= tx < GRID_W and 0 <= ty < GRID_H): return 
        if (tx, ty) in agents_map: return 
        
        del agents_map[(self.x, self.y)]
        self.x, self.y = tx, ty
        agents_map[(self.x, self.y)] = self
        
        # 吃草 (只能维持生命，不能繁殖)
        if (self.x, self.y) in foods:
            foods.remove((self.x, self.y))
            self.energy += ENERGY_FOOD
            
        # 吃猛犸象 (必须合作)
        if (self.x, self.y) in mammoths:
            # 检查周围有没有"同色"队友
            has_partner = False
            my_tribe = self.signal > 0.5 # True是红族，False是蓝族
            
            for mx, my in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = self.x + mx, self.y + my
                if (nx, ny) in agents_map and agents_map[(nx, ny)] != self:
                    partner = agents_map[(nx, ny)]
                    partner_tribe = partner.signal > 0.5
                    
                    # [关键] 只有同族才能配合！
                    # 如果红蓝在一起，语言不通，配合失败
                    if my_tribe == partner_tribe:
                        has_partner = True
                        partner.energy += ENERGY_MAMMOTH / 2
                        break
            
            if has_partner:
                mammoths.remove((self.x, self.y))
                self.energy += ENERGY_MAMMOTH / 2
            else:
                pass # 配合失败，咬不动

        if self.energy <= 0:
            self.alive = False
            del agents_map[(self.x, self.y)]

# ==========================================
# 3. 运行
# ==========================================
def run_genesis_five():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genesis-Five: Tribal Wars")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    agents = []
    # 初始两族对立
    for _ in range(INIT_POPULATION):
        agents.append(Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)))

    foods = set()
    mammoths = set()
    agents_map = { (a.x, a.y): a for a in agents }
    
    def spawn_entity(target_set, count):
        for _ in range(count):
            fx, fy = random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)
            if (fx, fy) not in foods and (fx, fy) not in mammoths and (fx, fy) not in agents_map:
                target_set.add((fx, fy))

    spawn_entity(foods, 10)      # 草极少
    spawn_entity(mammoths, 50)   # 象很多

    running = True
    max_gen = 0
    
    while running:
        screen.fill(COLOR_BG)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        random.shuffle(agents)
        active_agents = [a for a in agents if a.alive]
        
        for a in active_agents:
            a.act(foods, mammoths, agents_map)
            
            # 繁殖
            if a.energy > 800:
                a.energy -= 400
                # 传递 parent_signal
                child = Agent(a.x, a.y, a.gen + 1, a.brain, parent_signal=a.signal)
                
                spawned = False
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = a.x + dx, a.y + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in agents_map:
                        child.x, child.y = nx, ny
                        agents.append(child)
                        agents_map[(nx, ny)] = child
                        spawned = True
                        if child.gen > max_gen: max_gen = child.gen
                        break
                if not spawned: a.energy += 400
        
        agents = [a for a in agents if a.alive]
        agents_map = { (a.x, a.y): a for a in agents }
        
        if len(foods) < 5: spawn_entity(foods, 1)
        if len(mammoths) < 50: spawn_entity(mammoths, 2)
        
        # 灭绝保护
        if len(agents) < 5:
            agents.append(Agent(random.randint(0, GRID_W-1), random.randint(0, GRID_H-1)))

        # 渲染
        for fx, fy in foods:
            pygame.draw.rect(screen, COLOR_FOOD, (fx*GRID_SIZE, fy*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for mx, my in mammoths:
            pygame.draw.rect(screen, COLOR_MAMMOTH, (mx*GRID_SIZE, my*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        for a in agents:
            # 鲜艳的红蓝显示
            if a.signal > 0.5:
                color = (255, 50, 50) # 红族
            else:
                color = (50, 50, 255) # 蓝族
                
            pygame.draw.rect(screen, color, (a.x*GRID_SIZE, a.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

        info = f"Pop: {len(agents)} | Gen: {max_gen}"
        screen.blit(font.render(info, True, (255, 255, 255)), (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_genesis_five()