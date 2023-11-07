class Matrix:
    def __init__(self, matrix=None):
        """
        Initializing matrix
        :param matrix: matrix data
        """
        if matrix is None:
            matrix = [[]]
        self.n = len(matrix)
        self.m = len(matrix[0])
        self.matrix = matrix

    def fromString(self, s, row):
        """
        Parses a string s and writes it to a matrix[row]
        :param s: string
        :param row: row number
        :return: matrix
        """
        self.matrix[row] = [int(i) for i in s.split()]
        return self

    def fromStringArray(self, s_arr):
        """
        Parses an array of strings and converts it to the matrix
        :param s_arr: array of strings
        :return: matrix
        """
        for row, s in enumerate(s_arr):
            self.fromString(s, row)
        return self

    def print(self, *args):
        """
        Prints the matrix
        :param args: rest of the thing to print
        :return: nothing
        """
        for row in range(self.n):
            print(['{:.6f}'.format(flt) for flt in self.matrix[row]])
        for arg in args:
            print(arg, end='')

    def inverse(self):
        """
        Inverse the matrix
        :return: matrix^-1
        """
        if self.n != self.m:
            raise ValueError("Matrix is not square, and thus not invertible.")

        identity = Matrix([[int(i == j) for j in range(self.n)] for i in range(self.n)])
        augmented_matrix = Matrix(
            [[self.matrix[i][j] for j in range(self.n)] + identity.matrix[i] for i in range(self.n)])

        # Perform Gaussian elimination with back-substitution
        for col in range(self.n):
            # Find the pivot row (the row with the largest absolute value in the current column)
            max_val = 0
            max_row = col
            for row in range(col, self.n):
                if abs(augmented_matrix.matrix[row][col]) > max_val:
                    max_val = abs(augmented_matrix.matrix[row][col])
                    max_row = row

            # Swap the current row with the pivot row
            augmented_matrix.matrix[col], augmented_matrix.matrix[max_row] = augmented_matrix.matrix[max_row], \
                augmented_matrix.matrix[col]

            # Scale the pivot row so that the pivot element becomes 1
            pivot_val = augmented_matrix.matrix[col][col]
            for j in range(col, 2 * self.n):
                augmented_matrix.matrix[col][j] /= pivot_val

            # Eliminate other rows
            for i in range(self.n):
                if i != col:
                    factor = augmented_matrix.matrix[i][col]
                    for j in range(col, 2 * self.n):
                        augmented_matrix.matrix[i][j] -= factor * augmented_matrix.matrix[col][j]

        # Extract the right half of the augmented matrix, which is the inverse
        inverse_matrix = Matrix([augmented_matrix.matrix[i][self.n:] for i in range(self.n)])

        return inverse_matrix

    def transpose(self):
        """
        Transposes the matrix
        :return: matrix^T
        """
        transposed = [[self.matrix[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(transposed)

    def add(self, other):
        """
        Adds other matrix to this
        :param other: matrix to add
        :return: matrix + other
        """
        if self.n != other.n or self.m != other.m:
            raise ValueError("Matrix dimensions are not compatible for addition.")

        result = [[0 for _ in range(self.m)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                result[i][j] = self.matrix[i][j] + other.matrix[i][j]

        return Matrix(result)

    def subtract(self, other):
        """
        Subtracts other matrix to this
        :param other: matrix to add
        :return: matrix - other
        """
        return self.add(other.multiply(-1))

    def multiply(self, other):
        """
        Multiplies other matrix to this
        :param other: integer or matrix
        :return: matrix * other
        """
        def multiply_matrix(self, other):
            if self.m != other.n:
                raise ValueError("Matrix dimensions are not compatible for multiplication.")

            result = [[0 for _ in range(other.m)] for _ in range(self.n)]
            for i in range(self.n):
                for j in range(other.m):
                    for k in range(self.m):
                        result[i][j] += self.matrix[i][k] * other.matrix[k][j]

            return Matrix(result)

        def multiply_constant(self, other):
            result = [[0 for _ in range(self.m)] for _ in range(self.n)]
            for i in range(self.n):
                for j in range(self.m):
                    result[i][j] = self.matrix[i][j] * other

            return Matrix(result)

        if type(other) == Matrix:
            return multiply_matrix(self, other)
        else:
            return multiply_constant(self, other)

    def min(self):
        """
        Returns minimal element in all the matrix
        :return: minimal element
        """
        ans = None
        for i in range(self.n):
            for j in range(self.m):
                ans = min(ans, self.matrix[i][j]) if (i != 0 or j != 0) else self.matrix[i][j]
        return ans


class Vector(Matrix):
    def __init__(self, matrix=None):
        super().__init__(matrix)
        if matrix is None:
            matrix = [[]]
        self.n = 1
        self.m = len(matrix[0])

    def norm(self):
        """
        Calculates the norm of matrix
        :return: norm
        """
        norm = 0
        for num in self.matrix[0]:
            norm += num ** 2
        return norm ** 0.5

def I(n):
    """
    Identity matrix
    :param n: number of rows and columns
    :return: identity matrix
    """
    return Matrix([[(1 if i == j else 0) for i in range(n)] for j in range(n)])

class Interior:

    def __init__(self, a=Matrix(), b=Vector(), c=Vector(), eps=0.00001, alpha=0.5):
        self.a = a
        self.b = b
        self.c = c
        self.eps = eps
        self.alpha = alpha

    def random_point(self):
        """
        Gets the random point which satisfies the system of inequalities
        :return: vector of points x
        """
        n, m = self.a.m, self.a.n
        if n < m:
            raise ValueError("No solution found within the maximum iterations")
        x = [1 for _ in range(n - m)]
        for i in range(m):
            x.append((self.b.matrix[i][0] -
                      (sum([j for j in self.a.matrix[i]]) - self.a.matrix[i][i + n - m])) / self.a.matrix[i][i + n - m])
        return Vector([x]).transpose()

    def steps(self, x=None):
        """
        Interior-Point algorithm
        :param x: vector x
        :return: new vector x
        """
        if x is None:
            x = self.random_point()

        D = Matrix([[(x.matrix[i][0] if i == j else 0) for i in range(x.n)] for j in range(x.n)])
        A_new = self.a.multiply(D)

        C_new = D.multiply(self.c.transpose())
        A_new_t = A_new.transpose()
        inverse = A_new.multiply(A_new_t).inverse()
        A_final = A_new_t.multiply(inverse).multiply(A_new)

        P = I(A_final.n).subtract(A_final)
        C_p = P.multiply(C_new)
        v = C_p.min()
        x_new = Vector([[1 for _ in range(A_final.n)]]).transpose().add(C_p.multiply(self.alpha / abs(v)))
        x = D.multiply(x_new)
        return x


A = Matrix([[2, 4, 1, 0], [1, 3, 0, -1]])
B = Vector([[16, 9]]).transpose()
C = Vector([[-1, -1, 0, 0]])

Inter = Interior(A, B, C)
X = Vector([[0.5, 3.5, 1, 2]]).transpose()

iteration_count = 0
while True:
    iteration_count += 1
    X_prev = X
    X = Inter.steps(X)
    if Inter.eps > Vector(X.transpose().subtract(X_prev.transpose()).matrix).norm():
        print("Final iteration:")
        X.print()
        break
    else:
        print(f"Iteration #{iteration_count}:")
        X.print('\n')
