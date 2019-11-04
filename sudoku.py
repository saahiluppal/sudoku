import numpy as np
# Finding unsige cell
# File name: SUDOKU.py


def FindUnsignedLocation(Board, l):
    for row in range(0, 9):
        for col in range(0, 9):
            if (Board[row][col] == 0):
                l[0] = row
                l[1] = col
                return True
    return False

# Hàm kiểm tra tính an toàn của những ô trong hàng


def InRow(Board, row, num):
    for i in range(0, 9):
        if (Board[row][i] == num):
            return True
    return False

# Hàm kiểm tra tính an toàn của những ô trong cột


def InCol(Board, col, num):
    for i in range(0, 9):
        if (Board[i][col] == num):
            return True
    return False

# Hàm kiểm tra tính an toàn của các ô trong một ô lớn 3x3


def InBox(Board, row, col, num):
    for i in range(0, 3):
        for j in range(0, 3):
            if (Board[i + row][j + col] == num):
                return True
    return False

# Kiểm tra trạng thái an toàn tại một vị trí


def isSafe(Board, row, col, num):
    return not InCol(Board, col, num) and not InRow(Board, row, num) and not InBox(Board, row - row % 3, col - col % 3, num)


def SolveSudoku(Board):
    l = [0, 0]
    if (not FindUnsignedLocation(Board, l)):
        return True
    row = l[0]
    col = l[1]
    for num in range(1, 10):
        if (isSafe(Board, row, col, num)):
            Board[row][col] = num
            if (SolveSudoku(Board)):
                print(Board)
                break
            Board[row][col] = 0
    return False


Board = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
         [6, 0, 0, 1, 9, 5, 0, 0, 0],
         [0, 9, 8, 0, 0, 0, 0, 6, 0],
         [8, 0, 0, 0, 6, 0, 0, 0, 3],
         [4, 0, 0, 8, 0, 3, 0, 0, 1],
         [7, 0, 0, 0, 2, 0, 0, 0, 6],
         [0, 6, 0, 0, 0, 0, 2, 8, 0],
         [0, 0, 0, 4, 1, 9, 0, 0, 5],
         [0, 0, 0, 0, 8, 0, 0, 7, 9]]

#SolveSudoku(Board)
