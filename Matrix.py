import numpy as np
from collections.abc import Iterable
import pprint as pp

class Matrix:
    def __init__(self, shape=None, asArr=None):
        assert shape is not None or asArr is not None
        self.shape = None
        self.mat = None
        if shape is not None:
            assert isinstance(shape, tuple)
            self.shape = shape
            assert len(shape) == 2
            self.mat = np.zeros(shape)
        if asArr is not None:
            assert isinstance(asArr, Iterable)
            if not isinstance(asArr, np.ndarray):
                self.mat = np.array(asArr)
                self.shape = self.mat.shape
            else:
                self.mat = asArr
                self.shape = self.mat.shape

    def __getitem__(self, key):
        return self.mat[key]
    def __str__(self):
        return str(self.mat)
    def __unicode__(self):
        return str(self.mat)
    def __repr__(self):
        return str(self.mat)

    @staticmethod
    def ColumnVector(lst):
        return Matrix(asArr=[[_] for _ in lst])

    @staticmethod
    def I(dim):
        mat = Matrix(shape=(dim, dim))
        for i in range(dim):
            mat[i][i] = 1
        return mat

    @staticmethod
    def Diagonal(vector):
        mat = Matrix(shape=(len(vector), len(vector)))
        for i in range(len(vector)):
            mat[i][i] = vector[i]
        return mat

    def Mul(self, other):
        assert isinstance(other, Matrix)
        return Matrix(asArr=np.matmul(self.mat, other.mat))

    def Translate(self, direction):
        transMat = Matrix.I(len(direction) + 1)
        for i, val in enumerate(direction):
            transMat[i][len(direction)] = val
        beforeV = Matrix(asArr=[x[0] for x in self.mat] + [1])
        afterV = transMat.Mul(beforeV)
        self.mat = Matrix.ColumnVector(afterV[:-1]).mat

    def Rotate(self, rotationVector):
        assert self.shape[0] == 3 and self.shape[1] == 1
        xTheta, yTheta, zTheta = map(np.radians, rotationVector)
        Rx = Matrix(asArr=[
            [1, 0, 0],
            [0, np.cos(xTheta), -np.sin(xTheta)],
            [0, np.sin(xTheta), np.cos(xTheta)]])
        Ry = Matrix(asArr=[
            [np.cos(yTheta), 0, np.sin(yTheta)],
            [0, 1, 0],
            [-np.sin(yTheta), 0, np.cos(yTheta)]])
        Rz = Matrix(asArr=[
            [np.cos(zTheta), -np.sin(zTheta), 0],
            [np.sin(zTheta), np.cos(zTheta), 0],
            [0, 0, 1]])
        self.mat = Rz.Mul(Ry).Mul(Rx).Mul(self)

    def Scale(self, scaleVector):
        self.mat = Matrix.Diagonal(scaleVector).Mul(self)


def Test():
    np.set_printoptions(precision=3)
    print('identity\n', Matrix.I(4))
    print('arbitrary\n', Matrix(asArr=[[1,2,3], [1,2,3], [1,2,3]]))
    mat = Matrix.ColumnVector([1, 0, 0])
    mat.Rotate((0, 45, 0))
    print('rotate\n', mat)
    mat = Matrix.ColumnVector((1, 1, 1))
    mat.Translate((1, 0, 0))
    print("translate\n", mat)
    mat = Matrix.ColumnVector((4, 5, 6))
    mat.Scale((1, 2, 3))
    print("scale\n", mat)


if __name__ == '__main__':
    Test()