from Matrix import *
from main import *
import math
from MyFloat import MyFloat
PTWO=MyFloat(2)
NTWO=MyFloat(-2)
PONE=MyFloat(1)
NONE=MyFloat(-1)
ZERO=MyFloat(0)

class Camera:
    def __init__(self):
        self.mat = Matrix(shape=(4, 4))
        self.style = None
        self.pos = Point(0, 0, 0)

    def ConfigureOrthographic(self, left, right, bottom, top, near, far, manual=True):
        if manual:
            left = MyFloat(left)
            right = MyFloat(right)
            bottom = MyFloat(bottom)
            top = MyFloat(top)
            near = MyFloat(near)
            far = MyFloat(far)
            self.mat[0][0] = (PTWO / (right - left)).original
            self.mat[1][1] = (PTWO / (top - bottom)).original
            self.mat[2][2] = (NTWO / (far - near)).original
            self.mat[0][3] = (-(right + left) / (right - left)).original
            self.mat[0][3] = (-(top + bottom) / (top - bottom)).original
            self.mat[0][3] = (-(far + near) / (far - near)).original
            self.mat[3][3] = (PONE).original
        else:
            self.mat[0][0] = 2 / (right - left)
            self.mat[1][1] = 2 / (top - bottom)
            self.mat[2][2] = -2 / (far - near)
            self.mat[0][3] = -(right + left) / (right - left)
            self.mat[0][3] = -(top + bottom) / (top - bottom)
            self.mat[0][3] = -(far + near) / (far - near)
            self.mat[3][3] = 1
        self.style = 'Orthographic'

    def ConfigurePerspective(self, fov, near, far, manual=True):
        S = 1 / (math.tan(np.radians(fov/2)))
        self.mat[0][0] = S
        self.mat[1][1] = S
        self.mat[2][2] = -far / (far - near)
        self.mat[2][3] = -1
        self.mat[3][2] = -far * near / (far - near)
        self.mat[3][3] = 0
        self.style = 'Perspective'

    def ToCameraPoint(self, point):
        adjustedPoint = Point(*point.AsTuple())
        adjustedPoint.Translate(self.pos.AsTuple())
        mat = Matrix.ColumnVector(list(adjustedPoint.AsTuple()) + [1])
        cameraPoint = self.mat.Mul(mat)
        return cameraPoint[0][0], cameraPoint[1][0]