"""
八皇后问题求解器
采用回溯算法解决 N 皇后问题，支持任意正整数 N
"""

def solve_n_queens(n: int) -> list[list[str]]:
    """
    求解 N 皇后问题，返回所有合法的棋盘布局
    
    参数:
        n: 棋盘大小（皇后数量）
    
    返回:
        所有合法布局的列表，每个布局是字符串列表，例如：
        [
            [".Q..", "...Q", "Q...", "..Q."],
            ["..Q.", "Q...", "...Q", ".Q.."]
        ]
    """
    result = []  # 存储最终结果
    
    def backtrack(
        row: int, 
        cols: set, 
        diag1: set, 
        diag2: set, 
        path: list[int]
    ):
        """
        回溯函数，逐行放置皇后
        
        参数:
            row: 当前处理的行号
            cols: 已占用的列集合
            diag1: 已占用的主对角线（行-列）集合
            diag2: 已占用的副对角线（行+列）集合
            path: 记录每行皇后的列位置
        """
        # 所有行都放置完成，生成结果
        if row == n:
            board = []
            for col in path:
                # 生成每行的棋盘字符串（Q 表示皇后，. 表示空）
                board.append("." * col + "Q" + "." * (n - col - 1))
            result.append(board)
            return
        
        # 遍历当前行的所有列，尝试放置皇后
        for col in range(n):
            # 检查列、主对角线、副对角线是否冲突
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                # 递归处理下一行，更新占用的列和对角线
                backtrack(
                    row + 1,
                    cols | {col},          # 新增列
                    diag1 | {row - col},   # 新增主对角线
                    diag2 | {row + col},   # 新增副对角线
                    path + [col]           # 记录当前列位置
                )
    
    # 初始化回溯函数
    backtrack(0, set(), set(), set(), [])
    return result

# 测试代码（可选）
if __name__ == "__main__":
    # 测试 4 皇后
    print("4 皇后解的数量:", len(solve_n_queens(4)))
    # 测试 8 皇后
    print("8 皇后解的数量:", len(solve_n_queens(8)))
