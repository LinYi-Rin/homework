"""
八皇后问题单元测试
使用 pytest 框架验证求解器的正确性
"""
import sys
import os
# 添加 src 目录到 Python 路径，确保能导入 queens 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from queens import solve_n_queens

def test_n_queens_4():
    """测试 4 皇后问题（预期 2 个解）"""
    solutions = solve_n_queens(4)
    assert len(solutions) == 2, "4 皇后问题应该有 2 个解"

def test_n_queens_8():
    """测试 8 皇后问题（预期 92 个解）"""
    solutions = solve_n_queens(8)
    assert len(solutions) == 92, "8 皇后问题应该有 92 个解"

def test_n_queens_1():
    """测试 1 皇后问题（边界情况，预期 1 个解）"""
    solutions = solve_n_queens(1)
    assert len(solutions) == 1
    assert solutions[0] == ["Q"]

def test_n_queens_2():
    """测试 2 皇后问题（无解）"""
    solutions = solve_n_queens(2)
    assert len(solutions) == 0
