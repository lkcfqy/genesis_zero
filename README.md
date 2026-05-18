# genesis_zero

人工生命与神经演化实验合集。仓库通过一系列 `genesis_*.py` 脚本，从最小 XOR 动态网络开始，逐步扩展到能量、繁殖、捕食、循环大脑、环境平衡、部落信号和合作狩猎等模拟。

## 当前状态

这是探索性可视化 sandbox，不是统一框架。每个脚本都是一个相对独立的实验阶段，适合按顺序运行、观察规则变化对群体行为的影响。

仓库中已经包含两张示例图：`genesis_one_chart.png` 和 `evolution_chart.png`。

## 脚本说明

- `main.py`：动态神经网络尝试演化解决 XOR。
- `genesis_one.py`：带能量和繁殖的简单个体演化。
- `genesis_two.py`：Pygame 网格世界，加入食物和猎手。
- `genesis_three.py`：GRU 风格循环大脑与暗森林互动。
- `genesis_four.py`：大脑结构突变和环境平衡实验。
- `genesis_five.py`：信号、部落、草、猛犸象与合作狩猎。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pygame
```

运行示例：

```bash
python main.py
python genesis_one.py
python genesis_two.py
python genesis_three.py
python genesis_four.py
python genesis_five.py
```

`genesis_two.py` 之后的脚本需要图形界面显示 Pygame 窗口。

## 注意事项

- 多数参数直接写在脚本顶部，适合快速调参观察。
- 模拟结果依赖随机初始化，不同运行之间会有明显差异。
- 没有独立 `requirements.txt`，依赖需按上面的最小列表手动安装。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
