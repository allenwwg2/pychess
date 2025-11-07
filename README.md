# pychess
# 命令行中国象棋AI游戏 - 完整技术文档

## 项目概述

这是一个功能完整的命令行中国象棋AI游戏，采用Python实现，包含了完整的中国象棋规则引擎和基于Alpha-Beta剪枝的智能AI对战系统。

## 目录结构

- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [AI算法详解](#ai算法详解)
- [使用指南](#使用指南)
- [代码结构分析](#代码结构分析)
- [扩展开发](#扩展开发)

## 核心功能

### 🎮 完整的游戏规则实现

| 功能模块 | 描述 |
|---------|------|
| **棋盘表示** | 10行×9列的标准中国象棋棋盘 |
| **棋子走法** | 完整实现所有7种棋子的走法规则 |
| **特殊规则** | 马腿、象眼、炮的吃子、兵卒过河等 |
| **将军检测** | 实时检测将军状态和应将走法 |
| **将帅照面** | 正确处理将帅不能直接对面的规则 |
| **胜负判定** | 自动判断将死和游戏结束条件 |

### 🖥️ 用户界面特性

| 特性 | 描述 |
|------|------|
| **双显示模式** | 支持中文和英文棋子显示 |
| **走法高亮** | 高亮显示上一步移动的棋子 |
| **彩色输出** | 使用ANSI颜色代码区分红黑双方 |
| **实时状态** | 显示当前回合、上一步移动等信息 |

### 💾 游戏管理功能

| 功能 | 描述 |
|------|------|
| **存档系统** | 使用pickle进行游戏保存和加载 |
| **命令系统** | 丰富的命令行交互命令 |
| **配置管理** | 实时调整AI参数和游戏设置 |

## 技术架构

### 核心类设计

```python
class ChineseChess:
    # 棋盘和游戏状态
    board: List[List[str]]          # 10×9棋盘数组
    current_player: str            # 当前玩家 ('red'/'black')
    move_history: List[Tuple]      # 走法历史记录
    last_move: Optional[Tuple]     # 最后一次移动
    
    # AI配置
    ai_enabled: bool               # AI开关
    search_depth: int              # 搜索深度
    thinking_time: float           # 思考时间限制
    use_cache: bool               # 置换表缓存开关
    
    # 性能优化
    transposition_table: Dict      # 置换表
    zobrist_table: Dict           # Zobrist哈希表
    nodes_searched: int           # 搜索节点统计
```

### 主要方法分类

#### 1. 游戏核心方法
- `init_board()` - 初始化棋盘
- `make_move()` - 执行走法
- `undo_move()` - 撤销走法
- `is_game_over()` - 游戏结束判断

#### 2. 规则验证方法
- `is_valid_move()` - 基本走法验证
- `get_all_legal_moves()` - 获取所有合法走法
- `is_in_check()` - 将军检测
- `kings_face_each_other_after_move()` - 将帅照面检查

#### 3. AI搜索方法
- `alpha_beta_search()` - Alpha-Beta剪枝搜索
- `iterative_deepening_search()` - 迭代加深搜索
- `order_moves()` - 走法排序

#### 4. 评估方法
- `evaluate_board()` - 局面评估函数
- `get_piece_values()` - 棋子价值表
- `get_position_value()` - 位置价值计算

## AI算法详解

### 搜索算法架构

```
迭代加深搜索 (Iterative Deepening)
    ↓
Alpha-Beta剪枝 (Alpha-Beta Pruning)
    ↓
走法排序 (Move Ordering)
    ↓
置换表缓存 (Transposition Table)
    ↓
局面评估 (Position Evaluation)
```

### Alpha-Beta剪枝算法

```python
def alpha_beta_search(self, depth, alpha, beta, maximizing_player, start_time):
    # 终止条件
    if depth == 0 or time_out or game_over:
        return evaluate_board()
    
    if maximizing_player:
        max_eval = -∞
        for move in ordered_moves:
            make_move(move)
            eval = alpha_beta_search(depth-1, alpha, beta, False, start_time)
            undo_move(move)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # Beta剪枝
                break
        return max_eval
    else:
        # 类似的minimizing逻辑...
```

### 评估函数设计

#### 子力价值表
```python
PIECE_VALUES = {
    'K': 10000,  # 将/帅
    'R': 500,    # 车
    'H': 300,    # 马  
    'C': 300,    # 炮
    'B': 200,    # 相/象
    'A': 200,    # 仕/士
    'P': 100     # 兵/卒
}
```

#### 评估公式
```
总评估值 = 红方子力价值 - 黑方子力价值
          + 红方位置价值 - 黑方位置价值
          + 特殊局面评估（将军、将死等）
```

### 性能优化技术

#### 1. Zobrist哈希
- 为每个(棋子,位置)对生成随机64位整数
- 通过异或运算快速计算棋盘哈希值
- 支持置换表去重

#### 2. 置换表
```python
transposition_table = {
    zobrist_key: {
        'depth': 搜索深度,
        'value': 评估值
    }
}
```

#### 3. 历史启发式
- 吃子走法优先搜索
- 根据历史得分排序走法
- 提高Alpha-Beta剪枝效率

## 使用指南

### 快速开始

```bash
# 运行游戏
python pychess.py

# 基本操作示例
# 输入走法: 起始行 起始列 目标行 目标列
7 1 0 1    # 移动黑方炮

# 常用命令
help        # 查看帮助
restart     # 重新开始
mode chinese # 切换中文显示
```

### 完整命令参考

#### 游戏控制命令
| 命令 | 语法 | 描述 |
|------|------|------|
| 重新开始 | `restart` | 重置游戏状态 |
| 退出游戏 | `quit`/`exit` | 退出程序 |
| 保存游戏 | `save <文件名>` | 保存当前游戏 |
| 加载游戏 | `load <文件名>` | 加载存档 |

#### 显示控制命令
| 命令 | 语法 | 描述 |
|------|------|------|
| 显示模式 | `mode chinese`/`mode english` | 切换棋子显示 |
| 局面评估 | `eval` | 显示当前评估分数 |
| 合法走法 | `moves` | 显示所有可行走法 |

#### AI配置命令
| 命令 | 语法 | 描述 |
|------|------|------|
| AI开关 | `ai on`/`ai off` | 启用/禁用AI |
| AI对战 | `ai vs ai` | AI自对战模式 |
| 搜索深度 | `depth <数值>` | 设置搜索深度(1-6) |
| 思考时间 | `time <秒数>` | 设置思考时间 |
| 缓存控制 | `cache on`/`cache off` | 置换表缓存控制 |

### 走法输入格式

```
起始行 起始列 目标行 目标列

示例:
7 1 0 1    # 从(7,1)移动到(0,1)
3 0 4 0    # 从(3,0)移动到(4,0)
```

**坐标系统**:
- 行号: 0-9 (0=红方底线, 9=黑方底线)
- 列号: 0-8 (从左到右)

## 代码结构分析

### 棋子走法生成器

#### 将/帅 (`get_king_moves`)
```python
def get_king_moves(self, row, col):
    directions = [(0,1), (1,0), (0,-1), (-1,0)]  # 上下左右
    # 限制在九宫格内移动
    # 检查将帅照面规则
```

#### 车 (`get_rook_moves`)  
```python
def get_rook_moves(self, row, col):
    # 直线移动，直到遇到棋子或边界
    # 可以吃对方棋子，不能越过棋子
```

#### 马 (`get_knight_moves`)
```python
def get_knight_moves(self, row, col):
    knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), 
                   (1,-2), (1,2), (2,-1), (2,1)]
    # 检查马腿阻碍
```

#### 炮 (`get_cannon_moves`)
```python
def get_cannon_moves(self, row, col):
    # 移动时: 直线移动，不能越过棋子
    # 吃子时: 必须跳过恰好一个棋子
```

#### 兵/卒 (`get_pawn_moves`)
```python
def get_pawn_moves(self, row, col):
    # 红兵: 向下移动，过河后可左右移动
    # 黑卒: 向上移动，过河后可左右移动
```

### 位置价值表系统

#### 兵/卒位置表
```python
def get_pawn_position_table(self):
    # 鼓励兵前进和占据中心位置
    # 价值随行数增加而提高
```

#### 马位置表  
```python
def get_knight_position_table(self):
    # 中心区域价值最高
    # 边角位置价值较低
    # 避免马被逼到棋盘边缘
```

### 游戏流程控制

```python
def play(self):
    while not game_over:
        display_board()
        
        if human_turn:
            handle_command(input())
        else:
            ai_move = iterative_deepening_search()
            make_move(ai_move)
```

## 扩展开发

### 算法优化方向

#### 1. 搜索算法改进
```python
# 实现MTD(f)算法
def mtdf_search(self, initial_depth):
    # 更高效的零窗口搜索
    pass

# 添加空着裁剪
def null_move_pruning(self):
    # 在适当情况下跳过回合加速搜索
    pass
```

#### 2. 评估函数增强
```python
def enhanced_evaluation(self):
    # 添加更多评估特征:
    # - 棋子灵活性
    # - 控制中心程度  
    # - 兵形结构
    # - 棋子协调性
    pass
```

#### 3. 开局库和残局库
```python
class OpeningBook:
    def __init__(self):
        self.opening_moves = load_opening_database()
    
    def get_book_move(self, position):
        # 返回开局库中的推荐走法
        pass
```

### 功能扩展建议

#### 1. 用户界面升级
- 添加图形界面 (PyGame/Tkinter)
- 实现走法动画效果
- 添加声音反馈

#### 2. 对战功能扩展
- 网络对战支持
- 棋谱记录和回放
- 等级分系统

#### 3. 分析工具
- 走法分析模式
- 胜负关键点标记
- 战术训练模式

### 性能调优参数

```python
# 可调整的AI参数
OPTIMAL_PARAMETERS = {
    'search_depth': 4,           # 平衡强度和速度
    'thinking_time': 3.0,        # 合理的思考时间
    'use_cache': True,           # 启用置换表
    'enable_null_move': True,    # 空着裁剪
    'aspiration_window': 50      # 期望搜索窗口
}
```

## 总结

这个中国象棋AI项目展示了传统博弈算法在现代程序设计中的应用，具有以下特点：

1. **完整性** - 实现了所有中国象棋规则
2. **智能性** - 基于Alpha-Beta剪枝的强AI对手
3. **可扩展性** - 模块化设计便于功能扩展
4. **教育价值** - 适合学习博弈算法和AI编程


