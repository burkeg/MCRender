from Matrix import *
from main import *

class Camera:
    def __init__(self):
        self.mat = Matrix(shape=(4, 4))
        self.style = None

    def ConfigureOrthographic(self, left, right, bottom, top, near, far):
        self.mat[0][0] = 2 / (right - left)
        self.mat[1][1] = 2 / (top - bottom)
        self.mat[2][2] = - 2 / (far - near)
        self.mat[0][3] = -(right + left) / (right - left)
        self.mat[0][3] = -(top + bottom) / (top - bottom)
        self.mat[0][3] = -(far + near) / (far - near)
        self.mat[3][3] = 1
        self.style = 'Orthographic'

    def ToCameraPoint(self, point):
        mat = Matrix.ColumnVector(list(point.AsTuple()) + [1])
        cameraPoint = self.mat.Mul(mat)
        return cameraPoint[0][0], cameraPoint[1][0]