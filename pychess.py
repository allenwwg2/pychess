import time
import random
import pickle
from collections import defaultdict


class ChineseChess:
    def __init__(self):
        # 棋盘表示 (10行×9列)
        self.board = [['.' for _ in range(9)] for _ in range(10)]
        self.current_player = 'red'  # 'red' 或 'black'
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.last_move = None  # 记录最后一次移动，用于高亮显示

        # AI设置
        self.ai_enabled = True
        self.search_depth = 3
        self.thinking_time = 5.0
        self.use_cache = True

        # 显示设置
        self.display_mode = 'chinese'  # 'chinese' 或 'english'

        # 性能统计
        self.nodes_searched = 0
        self.search_start_time = 0

        # Zobrist哈希表
        self.zobrist_table = self.init_zobrist_table()
        self.transposition_table = {}

        self.init_board()

    def init_zobrist_table(self):
        """初始化Zobrist哈希表"""
        random.seed(42)  # 固定种子保证可重现
        table = {}
        pieces = ['K', 'A', 'B', 'R', 'H', 'C', 'P',
                  'k', 'a', 'b', 'r', 'h', 'c', 'p', '.']
        for piece in pieces:
            table[piece] = [[random.getrandbits(64) for _ in range(9)] for _ in range(10)]
        return table

    def get_zobrist_key(self):
        """计算当前棋盘的Zobrist哈希值"""
        key = 0
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                key ^= self.zobrist_table[piece][i][j]
        return key

    def init_board(self):
        """初始化棋盘"""
        # 红方布局 (0-3行)
        self.board[0] = ['R', 'H', 'B', 'A', 'K', 'A', 'B', 'H', 'R']
        self.board[1] = ['.', '.', '.', '.', '.', '.', '.', '.', '.']
        self.board[2] = ['.', 'C', '.', '.', '.', '.', '.', 'C', '.']
        self.board[3] = ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P']
        self.board[4] = ['.', '.', '.', '.', '.', '.', '.', '.', '.']

        # 黑方布局 (6-9行)
        self.board[5] = ['.', '.', '.', '.', '.', '.', '.', '.', '.']
        self.board[6] = ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p']
        self.board[7] = ['.', 'c', '.', '.', '.', '.', '.', 'c', '.']
        self.board[8] = ['.', '.', '.', '.', '.', '.', '.', '.', '.']
        self.board[9] = ['r', 'h', 'b', 'a', 'k', 'a', 'b', 'h', 'r']

    def display_board(self):
        """显示棋盘，高亮显示最后一次移动的棋子"""
        chinese_pieces = {
            'K': '帅', 'A': '仕', 'B': '相', 'R': '车', 'H': '马', 'C': '炮', 'P': '兵',
            'k': '将', 'a': '士', 'b': '象', 'r': '車', 'h': '馬', 'c': '砲', 'p': '卒',
            '.': '・'
        }

        english_pieces = {
            'K': 'K', 'A': 'A', 'B': 'B', 'R': 'R', 'H': 'H', 'C': 'C', 'P': 'P',
            'k': 'k', 'a': 'a', 'b': 'b', 'r': 'r', 'h': 'h', 'c': 'c', 'p': 'p',
            '.': '.'
        }

        piece_map = chinese_pieces if self.display_mode == 'chinese' else english_pieces

        print("      0   1   2  3   4  5   6   7  8")
        print("     ───────────────────────────────")

        # 定义ANSI颜色代码
        try:
            # 如果终端支持颜色，设置颜色代码
            RED = '\033[91m'  # 红色 - 用于红方棋子
            BLACK = '\033[90m'  # 灰色 - 用于黑方棋子（纯黑可能看不清）
            HIGHLIGHT = '\033[103m\033[30m'  # 黄色背景 + 黑色文字 - 用于高亮显示
            RESET = '\033[0m'  # 重置颜色
            use_color = True  # 标记使用颜色
        except:
            # 如果终端不支持颜色，则不使用颜色
            use_color = False

        # 从第9行（棋盘顶部/黑方底线）到第0行（棋盘底部/红方底线）遍历每一行
        # 注意：这是为了符合传统棋盘显示习惯（黑方在上，红方在下）
        for i in range(10):
            # 确定当前行属于哪一方（楚河汉界划分）
            # 第5行及以上为黑方区域，第4行及以下为红方区域
            side = '黑' if i >= 5 else '红'

            # 只在棋盘顶部和底部显示方标识
            side_label = "黑方" if i == 9 else "红方" if i == 0 else ""

            # 存储当前行的显示内容
            row_display = []

            # 遍历当前行的每一列（0-8）
            for j in range(9):
                # 获取当前位置的棋子
                # 注意：这里直接使用 i 作为行索引是正确的
                # 因为 self.board 的行索引与显示行号一致
                # i=9 对应 self.board[9]（黑方底线）
                # i=0 对应 self.board[0]（红方底线）
                piece = self.board[i][j]
                # 根据显示模式获取棋子符号
                piece_char = piece_map[piece]

                # 确定棋子颜色
                if use_color:
                    if piece != '.':
                        # 根据棋子字符判断是红方还是黑方
                        if self.is_red_piece(piece):
                            color = RED  # 红方棋子用红色
                        else:
                            color = BLACK  # 黑方棋子用灰色
                    else:
                        color = ''  # 空位置不使用颜色
                else:
                    color = ''  # 不使用颜色

                # 检查是否是上一步移动的棋子位置
                # 注意：last_move 中存储的坐标也是基于相同的索引系统
                is_highlighted = False
                if self.last_move and ((i, j) == self.last_move[:2] or (i, j) == self.last_move[2:]):
                    is_highlighted = True
                    if self.display_mode == 'chinese':
                        display_char = f"[{piece_char}]"
                    else:
                        display_char = f"({piece_char})"
                else:
                    # 普通显示
                    if use_color:
                        display_char = f"{color} {piece_char} {RESET}"
                    else:
                        display_char = f" {piece_char} "

                # 将显示字符添加到行显示列表
                row_display.append(display_char)

            # 打印当前行：行号 + 方标识 + 棋子 + 方标识
            # 注意：这里显示的行号 i 与数组索引一致
            # 这样用户输入走法时可以直接使用显示的行号
            print(f"{i} {side}|{''.join(row_display)}| {side_label}")

        print("     ───────────────────────────────")
        print("      0   1   2  3   4  5   6   7  8")
        print(f"当前回合: {'红方' if self.current_player == 'red' else '黑方'}")

        # 显示最后一次移动信息
        if self.last_move:
            start_row, start_col, end_row, end_col = self.last_move
            start_piece = self.board[end_row][end_col]  # 注意：棋子现在在目标位置
            piece_name = chinese_pieces.get(start_piece, '未知')
            print(f"上一步移动: ({start_row}, {start_col}) -> ({end_row}, {end_col}) [{piece_name}]")

    def is_red_piece(self, piece):
        """判断是否为红方棋子"""
        return piece in 'KABRHCP'

    def is_black_piece(self, piece):
        """判断是否为黑方棋子"""
        return piece in 'kabrhcp'

    def get_all_legal_moves(self, player=None):
        """获取所有合法走法，优先考虑应将走法"""
        if player is None:
            player = self.current_player

        moves = []
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if (player == 'red' and piece.isupper()) or (player == 'black' and piece.islower()):
                    moves.extend(self.get_piece_moves(i, j))

        # 如果被将军，优先考虑应将走法
        if self.is_in_check(player):
            saving_moves = []
            for move in moves:
                if self.would_move_save_king(move, player):
                    saving_moves.append(move)
            return saving_moves

        return moves

    def is_in_check(self, player):
        """检查玩家是否被将军"""
        # 找到玩家的将/帅位置
        king_symbol = 'K' if player == 'red' else 'k'
        king_pos = None

        for i in range(10):
            for j in range(9):
                if self.board[i][j] == king_symbol:
                    king_pos = (i, j)
                    break
            if king_pos:
                break

        if not king_pos:
            return False  # 将/帅不在棋盘上，理论上不会发生

        # 检查对方是否有棋子可以攻击将/帅
        opponent = 'black' if player == 'red' else 'red'

        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if (opponent == 'red' and piece.isupper()) or (opponent == 'black' and piece.islower()):
                    # 检查这个棋子是否可以移动到将/帅位置
                    piece_moves = self.get_piece_moves(i, j)
                    if (king_pos[0], king_pos[1]) in [(move[2], move[3]) for move in piece_moves]:
                        return True

        return False

    def would_move_save_king(self, move, player):
        """检查走法是否能解除将军状态"""
        start_row, start_col, end_row, end_col = move

        # 执行走法
        captured_piece = self.make_move(start_row, start_col, end_row, end_col)

        # 检查是否仍然被将军
        still_in_check = self.is_in_check(player)

        # 撤销走法
        self.undo_move(start_row, start_col, end_row, end_col, captured_piece)

        return not still_in_check

    def get_piece_moves(self, row, col):
        """获取单个棋子的所有合法走法"""
        piece = self.board[row][col].upper()
        moves = []

        if piece == 'K':  # 将/帅
            moves = self.get_king_moves(row, col)
        elif piece == 'A':  # 仕/士
            moves = self.get_advisor_moves(row, col)
        elif piece == 'B':  # 相/象
            moves = self.get_bishop_moves(row, col)
        elif piece == 'R':  # 车
            moves = self.get_rook_moves(row, col)
        elif piece == 'H':  # 马
            moves = self.get_knight_moves(row, col)
        elif piece == 'C':  # 炮
            moves = self.get_cannon_moves(row, col)
        elif piece == 'P':  # 兵/卒
            moves = self.get_pawn_moves(row, col)

        return moves

    def get_king_moves(self, row, col):
        """将/帅的走法"""
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        piece = self.board[row][col]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(row, col, new_row, new_col):
                # 检查将帅照面
                if not self.kings_face_each_other_after_move(row, col, new_row, new_col):
                    # 检查是否在九宫格内
                    if piece.isupper():  # 红方帅
                        if 0 <= new_row <= 2 and 3 <= new_col <= 5:
                            moves.append((row, col, new_row, new_col))
                    else:  # 黑方将
                        if 7 <= new_row <= 9 and 3 <= new_col <= 5:
                            moves.append((row, col, new_row, new_col))

        return moves

    def get_advisor_moves(self, row, col):
        """仕/士的走法"""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        piece = self.board[row][col]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(row, col, new_row, new_col):
                # 检查是否在九宫格内
                if piece.isupper():  # 红方仕
                    if 0 <= new_row <= 2 and 3 <= new_col <= 5:
                        moves.append((row, col, new_row, new_col))
                else:  # 黑方士
                    if 7 <= new_row <= 9 and 3 <= new_col <= 5:
                        moves.append((row, col, new_row, new_col))

        return moves

    def get_bishop_moves(self, row, col):
        """相/象的走法"""
        moves = []
        directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        piece = self.board[row][col]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(row, col, new_row, new_col):
                # 检查象眼是否被塞
                eye_row, eye_col = row + dr // 2, col + dc // 2
                if self.board[eye_row][eye_col] != '.':
                    continue

                # 检查是否过河
                if piece.isupper():  # 红方相
                    if new_row >= 5:  # 不能过河
                        continue
                else:  # 黑方象
                    if new_row <= 4:  # 不能过河
                        continue

                moves.append((row, col, new_row, new_col))

        return moves

    def get_rook_moves(self, row, col):
        """车的走法"""
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dr, dc in directions:
            for step in range(1, 10):
                new_row, new_col = row + dr * step, col + dc * step
                if not (0 <= new_row < 10 and 0 <= new_col < 9):
                    break

                if not self.is_valid_move(row, col, new_row, new_col):
                    break

                moves.append((row, col, new_row, new_col))

                # 如果遇到棋子，停止这个方向
                if self.board[new_row][new_col] != '.':
                    break

        return moves

    def get_knight_moves(self, row, col):
        """马的走法"""
        moves = []
        # 马走"日"字，有8个可能方向
        knight_moves = [
            (-2, -1), (-2, 1),  # 向上走
            (-1, -2), (-1, 2),  # 向左走
            (1, -2), (1, 2),  # 向右走
            (2, -1), (2, 1)  # 向下走
        ]

        # 对应的马腿位置
        leg_positions = {
            (-2, -1): (-1, 0), (-2, 1): (-1, 0),  # 向上走的马腿在正上方
            (-1, -2): (0, -1), (-1, 2): (0, 1),  # 向左/右走的马腿在正左/右方
            (1, -2): (0, -1), (1, 2): (0, 1),  # 向左/右走的马腿在正左/右方
            (2, -1): (1, 0), (2, 1): (1, 0)  # 向下走的马腿在正下方
        }

        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if not (0 <= new_row < 10 and 0 <= new_col < 9):
                continue

            # 检查马腿
            leg_dr, leg_dc = leg_positions[(dr, dc)]
            leg_row, leg_col = row + leg_dr, col + leg_dc

            # 马腿必须在棋盘内
            if not (0 <= leg_row < 10 and 0 <= leg_col < 9):
                continue

            # 马腿位置有棋子则不能走
            if self.board[leg_row][leg_col] != '.':
                continue

            if self.is_valid_move(row, col, new_row, new_col):
                moves.append((row, col, new_row, new_col))

        return moves

    def get_cannon_moves(self, row, col):
        """炮的走法"""
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dr, dc in directions:
            has_piece = False
            for step in range(1, 10):
                new_row, new_col = row + dr * step, col + dc * step
                if not (0 <= new_row < 10 and 0 <= new_col < 9):
                    break

                if not has_piece:
                    # 炮移动时，路径上不能有棋子
                    if self.board[new_row][new_col] != '.':
                        has_piece = True
                        continue  # 跳过这个棋子，继续向后看
                    else:
                        if self.is_valid_move(row, col, new_row, new_col):
                            moves.append((row, col, new_row, new_col))
                else:
                    # 炮吃子时，必须跳过一个棋子
                    if self.board[new_row][new_col] != '.':
                        if self.is_valid_move(row, col, new_row, new_col):
                            moves.append((row, col, new_row, new_col))
                        break  # 吃子后停止

        return moves

    def get_pawn_moves(self, row, col):
        """兵/卒的走法"""
        moves = []
        piece = self.board[row][col]

        if piece.isupper():  # 红方兵
            # 红兵向前走 (向下)
            directions = [(1, 0)]
            # 过河后可以左右走
            if row >= 5:
                directions.extend([(0, 1), (0, -1)])
        else:  # 黑方卒
            # 黑卒向前走 (向上)
            directions = [(-1, 0)]
            # 过河后可以左右走
            if row <= 4:
                directions.extend([(0, 1), (0, -1)])

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(row, col, new_row, new_col):
                moves.append((row, col, new_row, new_col))

        return moves

    def is_valid_move(self, start_row, start_col, end_row, end_col):
        """验证走法是否基本合法"""
        # 检查是否在棋盘内
        if not (0 <= end_row < 10 and 0 <= end_col < 9):
            return False

        start_piece = self.board[start_row][start_col]
        end_piece = self.board[end_row][end_col]

        # 不能吃自己的棋子
        if (start_piece.isupper() and end_piece.isupper()) or \
                (start_piece.islower() and end_piece.islower()):
            return False

        return True

    def kings_face_each_other_after_move(self, start_row, start_col, end_row, end_col):
        """检查移动后是否将帅照面"""
        # 临时执行移动
        temp_piece = self.board[end_row][end_col]
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = '.'

        # 查找将帅位置
        king_red_pos = None
        king_black_pos = None

        for i in range(10):
            for j in range(9):
                if self.board[i][j] == 'K':
                    king_red_pos = (i, j)
                elif self.board[i][j] == 'k':
                    king_black_pos = (i, j)

        result = False
        if king_red_pos and king_black_pos and king_red_pos[1] == king_black_pos[1]:
            # 在同一列，检查中间是否有棋子
            col = king_red_pos[1]
            start_row_idx = min(king_red_pos[0], king_black_pos[0]) + 1
            end_row_idx = max(king_red_pos[0], king_black_pos[0])
            has_piece_between = False
            for row in range(start_row_idx, end_row_idx):
                if self.board[row][col] != '.':
                    has_piece_between = True
                    break
            result = not has_piece_between

        # 恢复棋盘
        self.board[start_row][start_col] = self.board[end_row][end_col]
        self.board[end_row][end_col] = temp_piece

        return result

    def alpha_beta_search(self, depth, alpha, beta, maximizing_player, start_time):
        """Alpha-Beta搜索算法"""
        self.nodes_searched += 1

        # 时间检查
        if time.time() - start_time > self.thinking_time:
            return self.evaluate_board()

        # 检查置换表
        zobrist_key = self.get_zobrist_key()
        if self.use_cache and zobrist_key in self.transposition_table:
            entry = self.transposition_table[zobrist_key]
            if entry['depth'] >= depth:
                return entry['value']

        # 叶子节点或游戏结束
        if depth == 0 or self.is_game_over():
            value = self.evaluate_board()
            if self.use_cache:
                self.transposition_table[zobrist_key] = {
                    'depth': depth,
                    'value': value
                }
            return value

        moves = self.get_all_legal_moves()
        if not moves:
            return self.evaluate_board()

        # 走法排序
        moves = self.order_moves(moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                start_row, start_col, end_row, end_col = move
                captured_piece = self.make_move(start_row, start_col, end_row, end_col)

                eval_score = self.alpha_beta_search(depth - 1, alpha, beta, False, start_time)
                self.undo_move(start_row, start_col, end_row, end_col, captured_piece)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta剪枝

            if self.use_cache:
                self.transposition_table[zobrist_key] = {
                    'depth': depth,
                    'value': max_eval
                }
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                start_row, start_col, end_row, end_col = move
                captured_piece = self.make_move(start_row, start_col, end_row, end_col)

                eval_score = self.alpha_beta_search(depth - 1, alpha, beta, True, start_time)
                self.undo_move(start_row, start_col, end_row, end_col, captured_piece)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha剪枝

            if self.use_cache:
                self.transposition_table[zobrist_key] = {
                    'depth': depth,
                    'value': min_eval
                }
            return min_eval

    def iterative_deepening_search(self):
        """迭代加深搜索"""
        self.nodes_searched = 0
        start_time = time.time()

        moves = self.get_all_legal_moves()
        if not moves:
            return None

        best_move = None
        best_value = float('-inf') if self.current_player == 'red' else float('inf')

        # 走法排序
        moves = self.order_moves(moves)

        for depth in range(1, self.search_depth + 1):
            current_best_move = None
            current_best_value = float('-inf') if self.current_player == 'red' else float('inf')

            for move in moves:
                start_row, start_col, end_row, end_col = move
                captured_piece = self.make_move(start_row, start_col, end_row, end_col)

                eval_score = self.alpha_beta_search(
                    depth - 1,
                    float('-inf'),
                    float('inf'),
                    self.current_player == 'black',
                    start_time
                )

                self.undo_move(start_row, start_col, end_row, end_col, captured_piece)

                # 时间检查
                if time.time() - start_time > self.thinking_time:
                    return best_move or moves[0]

                if (self.current_player == 'red' and eval_score > current_best_value) or \
                        (self.current_player == 'black' and eval_score < current_best_value):
                    current_best_value = eval_score
                    current_best_move = move

            if current_best_move:
                best_move = current_best_move
                best_value = current_best_value

            # 时间检查
            if time.time() - start_time > self.thinking_time:
                break

        search_time = time.time() - start_time

        print(f"AI思考时间: {search_time:.2f}秒")
        print(f"搜索深度: {depth}, 搜索节点: {self.nodes_searched}")

        return best_move

    def order_moves(self, moves):
        """走法排序：历史启发"""
        # 简单的吃子优先排序
        scored_moves = []
        for move in moves:
            score = 0
            start_row, start_col, end_row, end_col = move
            target_piece = self.board[end_row][end_col]

            # 吃子走法优先
            if target_piece != '.':
                # 根据被吃棋子价值评分
                piece_values = self.get_piece_values()
                target_value = piece_values.get(target_piece.upper(), 0)
                score += target_value

            scored_moves.append((score, move))

        # 按分数降序排序
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]

    def evaluate_board(self):
        """评估棋盘状态"""
        if self.is_checkmate('red'):
            return -100000  # 红方被将死
        if self.is_checkmate('black'):
            return 100000  # 黑方被将死

        score = 0

        # 子力价值评估
        piece_values = self.get_piece_values()

        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece != '.':
                    value = piece_values.get(piece.upper(), 0)
                    pos_value = self.get_position_value(piece, i, j)

                    if piece.isupper():  # 红方
                        score += value + pos_value
                    else:  # 黑方
                        score -= value + pos_value

        return score

    def get_piece_values(self):
        """棋子价值表"""
        return {
            'K': 10000, 'k': 10000,  # 将/帅
            'R': 500, 'r': 500,  # 车
            'H': 300, 'h': 300,  # 马
            'C': 300, 'c': 300,  # 砲/炮
            'B': 200, 'b': 200,  # 相/象
            'A': 200, 'a': 200,  # 仕/士
            'P': 100, 'p': 100  # 兵/卒
        }

    def get_position_value(self, piece, row, col):
        """获取棋子的位置价值"""
        position_tables = self.get_position_tables()
        piece_type = piece.upper()

        if piece_type in position_tables:
            table = position_tables[piece_type]
            if piece.isupper():  # 红方
                return table[row][col]
            else:  # 黑方
                # 黑方位置表是红方的镜像
                return table[9 - row][8 - col]

        return 0

    def get_position_tables(self):
        """返回所有棋子的位置价值表"""
        return {
            'P': self.get_pawn_position_table(),
            'H': self.get_knight_position_table(),
            'C': self.get_cannon_position_table(),
            'R': self.get_rook_position_table(),
            'A': self.get_advisor_position_table(),
            'B': self.get_bishop_position_table(),
            'K': self.get_king_position_table()
        }

    def get_pawn_position_table(self):
        """兵/卒的位置价值表"""
        # 红兵位置价值 - 鼓励兵前进和占据中心
        return [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 0, 10, 0, 15, 0, 10, 0, 10],
            [15, 0, 20, 0, 25, 0, 20, 0, 15],
            [20, 0, 25, 0, 30, 0, 25, 0, 20],
            [25, 0, 30, 0, 35, 0, 30, 0, 25],
            [30, 0, 35, 0, 40, 0, 35, 0, 30],
            [35, 0, 40, 0, 45, 0, 40, 0, 35],
            [40, 0, 45, 0, 50, 0, 45, 0, 40],
            [45, 0, 50, 0, 55, 0, 50, 0, 45]
        ]

    def get_knight_position_table(self):
        """马的位置价值表"""
        # 马在中心区域价值更高
        return [
            [-50, -40, -30, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 5, 5, 5, 0, -20, -40],
            [-30, 5, 10, 15, 15, 15, 10, 5, -30],
            [-30, 0, 15, 20, 20, 20, 15, 0, -30],
            [-30, 5, 15, 20, 20, 20, 15, 5, -30],
            [-30, 0, 10, 15, 15, 15, 10, 0, -30],
            [-30, 0, 0, 0, 0, 0, 0, 0, -30],
            [-40, -20, 0, 0, 0, 0, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -30, -40, -50],
            [-50, -40, -30, -30, -30, -30, -30, -40, -50]
        ]

    def get_cannon_position_table(self):
        """炮的位置价值表"""
        # 炮在河界和对方阵地价值更高
        return [
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [5, 5, 5, 10, 10, 10, 5, 5, 5],
            [5, 10, 10, 15, 15, 15, 10, 10, 5],
            [5, 10, 10, 15, 15, 15, 10, 10, 5],
            [5, 5, 5, 10, 10, 10, 5, 5, 5],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0]
        ]

    def get_rook_position_table(self):
        """车的位置价值表"""
        # 车在对方底线和开放线价值更高
        return [
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 0, 0, 0],
            [5, 5, 5, 10, 10, 10, 5, 5, 5],
            [10, 10, 10, 15, 15, 15, 10, 10, 10],
            [10, 10, 10, 15, 15, 15, 10, 10, 10],
            [5, 5, 5, 10, 10, 10, 5, 5, 5]
        ]

    def get_advisor_position_table(self):
        """仕/士的位置价值表"""
        # 士在九宫中心价值更高
        return [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 10, 0, 10, 0, 0, 0],
            [0, 0, 0, 0, 15, 0, 0, 0, 0],
            [0, 0, 0, 10, 0, 10, 0, 0, 0]
        ]

    def get_bishop_position_table(self):
        """相/象的位置价值表"""
        # 象在己方阵地价值稳定
        return [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 10, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 5, 0, 0]
        ]

    def get_king_position_table(self):
        """将/帅的位置价值表"""
        # 将在九宫中心价值更高
        return [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 10, 5, 0, 0, 0],
            [0, 0, 0, 10, 15, 10, 0, 0, 0],
            [0, 0, 0, 5, 10, 5, 0, 0, 0]
        ]

    def make_move(self, start_row, start_col, end_row, end_col):
        """执行走法"""
        captured_piece = self.board[end_row][end_col]
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = '.'

        self.move_history.append((start_row, start_col, end_row, end_col, captured_piece))
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        self.last_move = (start_row, start_col, end_row, end_col)  # 记录最后一次移动

        return captured_piece

    def undo_move(self, start_row, start_col, end_row, end_col, captured_piece):
        """撤销走法"""
        self.board[start_row][start_col] = self.board[end_row][end_col]
        self.board[end_row][end_col] = captured_piece
        self.move_history.pop()
        self.current_player = 'black' if self.current_player == 'red' else 'red'

        # 更新last_move为上一次移动
        if self.move_history:
            last_move = self.move_history[-1]
            self.last_move = (last_move[0], last_move[1], last_move[2], last_move[3])
        else:
            self.last_move = None

    def is_game_over(self):
        """检查游戏是否结束"""
        return self.is_checkmate('red') or self.is_checkmate('black')

    def is_checkmate(self, player):
        """检查是否被将死"""
        moves = self.get_all_legal_moves(player)
        return len(moves) == 0

    def handle_command(self, command):
        """处理命令行输入"""
        parts = command.strip().split()
        if not parts:
            return False

        cmd = parts[0].lower()

        if cmd in ['quit', 'exit']:
            return True

        elif cmd == 'restart':
            self.__init__()
            print("游戏已重新开始")

        elif cmd == 'save' and len(parts) > 1:
            self.save_game(parts[1])

        elif cmd == 'load' and len(parts) > 1:
            self.load_game(parts[1])

        elif cmd == 'mode' and len(parts) > 1:
            if parts[1] in ['chinese', 'english']:
                self.display_mode = parts[1]
                print(f"显示模式已切换为: {parts[1]}")
            else:
                print("无效的显示模式")

        elif cmd == 'eval':
            score = self.evaluate_board()
            print(f"当前局面评估(红方角度): {score}")

        elif cmd == 'moves':
            moves = self.get_all_legal_moves()
            print(f"合法走法数量: {len(moves)}")
            for i, move in enumerate(moves[:10]):  # 只显示前10个
                print(f"{i + 1}: {move}")
            if len(moves) > 10:
                print("...")

        elif cmd == 'ai':
            if len(parts) > 1:
                if parts[1] == 'on':
                    self.ai_enabled = True
                    print("AI已启用")
                elif parts[1] == 'off':
                    self.ai_enabled = False
                    print("AI已禁用")
                elif parts[1] == 'vs' and parts[2] == 'ai':
                    self.ai_vs_ai_mode()

        elif cmd == 'depth' and len(parts) > 1:
            try:
                self.search_depth = int(parts[1])
                print(f"AI搜索深度设置为: {self.search_depth}")
            except ValueError:
                print("无效的深度值")

        elif cmd == 'time' and len(parts) > 1:
            try:
                self.thinking_time = float(parts[1])
                print(f"AI思考时间设置为: {self.thinking_time}秒")
            except ValueError:
                print("无效的时间值")

        elif cmd == 'cache':
            if len(parts) > 1:
                if parts[1] == 'on':
                    self.use_cache = True
                    print("置换表缓存已启用")
                elif parts[1] == 'off':
                    self.use_cache = False
                    self.transposition_table.clear()
                    print("置换表缓存已禁用")

        elif cmd == 'help':
            self.show_help()

        else:
            # 尝试解析为走法
            try:
                if len(parts) == 4:
                    start_row, start_col, end_row, end_col = map(int, parts)
                    if self.make_human_move(start_row, start_col, end_row, end_col):
                        return False
                    else:
                        print("无效的走法")
                else:
                    print("无效命令")
            except ValueError:
                print("无效命令")

        return False

    def make_human_move(self, start_row, start_col, end_row, end_col):
        """执行人类玩家的走法"""
        # 验证走法合法性
        moves = self.get_all_legal_moves()
        if (start_row, start_col, end_row, end_col) in moves:
            self.make_move(start_row, start_col, end_row, end_col)
            return True
        return False

    def ai_vs_ai_mode(self):
        """AI对战AI模式"""
        print("开始AI对战AI模式")
        max_moves = 200  # 防止无限循环

        for move_count in range(max_moves):
            if self.is_game_over():
                break

            print(f"\n--- 第{move_count + 1}回合 ---")
            self.display_board()

            ai_move = self.iterative_deepening_search()
            if ai_move:
                start_row, start_col, end_row, end_col = ai_move
                self.make_move(start_row, start_col, end_row, end_col)
                print(f"AI走法: {start_row} {start_col} {end_row} {end_col}")
            else:
                print("无合法走法，游戏结束")
                break

            time.sleep(1)  # 暂停一下便于观察

        self.display_board()
        if self.is_checkmate('red'):
            print("黑方胜利! 红方被将死!")
        elif self.is_checkmate('black'):
            print("红方胜利! 黑方被将死!")
        else:
            print("和棋或达到最大回合数")

    def save_game(self, filename):
        """保存游戏"""
        try:
            with open(filename, 'wb') as f:
                game_data = {
                    'board': self.board,
                    'current_player': self.current_player,
                    'move_history': self.move_history,
                    'last_move': self.last_move,
                    'ai_enabled': self.ai_enabled,
                    'search_depth': self.search_depth,
                    'thinking_time': self.thinking_time,
                    'use_cache': self.use_cache,
                    'display_mode': self.display_mode
                }
                pickle.dump(game_data, f)
            print(f"游戏已保存到: {filename}")
        except Exception as e:
            print(f"保存失败: {e}")

    def load_game(self, filename):
        """加载游戏"""
        try:
            with open(filename, 'rb') as f:
                game_data = pickle.load(f)
                self.board = game_data['board']
                self.current_player = game_data['current_player']
                self.move_history = game_data['move_history']
                self.last_move = game_data.get('last_move')
                self.ai_enabled = game_data.get('ai_enabled', True)
                self.search_depth = game_data.get('search_depth', 3)
                self.thinking_time = game_data.get('thinking_time', 5.0)
                self.use_cache = game_data.get('use_cache', True)
                self.display_mode = game_data.get('display_mode', 'chinese')
            print(f"游戏已从 {filename} 加载")
        except Exception as e:
            print(f"加载失败: {e}")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
游戏控制:
  restart        - 重新开始游戏
  quit/exit      - 退出游戏
  save <file>    - 保存游戏
  load <file>    - 加载游戏

显示控制:
  mode chinese   - 中文显示模式
  mode english   - 英文显示模式
  eval           - 显示当前评估值
  moves          - 显示所有合法走法

AI控制:
  ai on/off      - 开启/关闭AI对战
  ai vs ai       - AI对战AI模式
  depth <n>      - 设置AI搜索深度
  time <秒>      - 设置AI思考时间
  cache on/off   - 开启/关闭置换表缓存

走法输入:
  <起始行> <起始列> <目标行> <目标列>
  示例: 7 1 0 1

帮助:
  help           - 显示此帮助信息
        """
        print(help_text)

    def play(self):
        """主游戏循环"""
        print("欢迎来到命令行中国象棋AI!")
        print("输入 'help' 查看可用命令")

        while True:
            self.display_board()

            # 检查游戏是否结束
            if self.is_game_over():
                if self.is_checkmate('red'):
                    print("黑方胜利! 红方被将死!")
                elif self.is_checkmate('black'):
                    print("红方胜利! 黑方被将死!")
                break

            if self.current_player == 'red' or not self.ai_enabled:
                # 人类玩家回合
                command = input("\n请输入命令或走法: ")
                if self.handle_command(command):
                    break
            else:
                # AI回合
                print("AI思考中...")
                ai_move = self.iterative_deepening_search()
                if ai_move:
                    start_row, start_col, end_row, end_col = ai_move
                    self.make_move(start_row, start_col, end_row, end_col)
                    print(f"AI走法: {start_row} {start_col} {end_row} {end_col}")
                else:
                    print("AI无法找到合法走法，游戏结束")
                    break


if __name__ == "__main__":
    game = ChineseChess()
    game.play()