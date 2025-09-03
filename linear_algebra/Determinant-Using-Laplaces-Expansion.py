def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Base case: 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    total = 0
    for j in range(len(matrix)):
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        c = ((j % 2) * - 2 + 1)
        cofactor = c * matrix[0][j] * determinant_4x4(submatrix)
        total += cofactor

    return total