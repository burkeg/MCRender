import numpy as np
import math
import struct
from enum import Enum
# http://weitz.de/ieee/
# https://ieeexplore.ieee.org/abstract/document/4380621/authors#authors

class FloatFlag(Enum):
    OK = 0
    INVALID = 1 << 0
    DIV0 = 1 << 1
    OVERFLOW = 1 << 2
    UNDERFLOW = 1 << 3
    INEXACT = 1 << 4

class MyFloat:
    status = FloatFlag.OK

    def __init__(self, value=None, valueFormat=None):
        self.format = None
        self.original = 0.0
        if value is not None:
            if valueFormat is not None:
                self.format = valueFormat
                self.original = valueFormat(value)
            else:
                self.format = type(value)
                self.original = value
        else:
            assert valueFormat is not None
            self.format = valueFormat
            self.original = valueFormat(0.0)
        self.CommonName = None
        self.Base = None
        self.SignificandBits = None
        self.DecimalDigits = None
        self.ExponentBits = None
        self.DecimalEMax = None
        self.ExponentBias = None
        self.EMin = None
        self.EMax = None
        self.S = None
        self.E = None
        self.T = None
        self.PopulateData()

    def __str__(self):
        return str(self.original) + ': ' + '{' + ','.join([self.S, self.E, self.T]) + '}'
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return str(self)
    def __add__(self, other):
        return MyFloat.Add(self, other)
    def __radd__(self, other):
        return MyFloat.Add(self, other)
    def __mul__(self, other):
        return MyFloat.Multiply(self, other)
    def __rmul__(self, other):
        return MyFloat.Multiply(self, other)

    @staticmethod
    def SetFlag(flag):
        MyFloat.status |= flag

    @staticmethod
    def ClearFlag(flag):
        MyFloat.status &= ~flag

    def Reinterpret(self):
        self.original = self.bin2float(self.S + self.E + self.T)
        pass

    def AsBinary(self):
        return self.float2bin(self.original)

    # https://stackoverflow.com/a/59594903
    def bin2float(self, b):
        ''' Convert binary string to a float.

        Attributes:
            :b: Binary string to transform.
        '''
        if self.format == np.float16:
            h = int(b, 2).to_bytes(2, byteorder="big")
            return struct.unpack('>e', h)[0]
        elif self.format == np.float32:
            h = int(b, 2).to_bytes(4, byteorder="big")
            return struct.unpack('>f', h)[0]
        elif self.format == np.float64:
            h = int(b, 2).to_bytes(8, byteorder="big")
            return struct.unpack('>d', h)[0]

    def float2bin(self, f):
        ''' Convert float to n-bit binary string.

        Attributes:
            :f: Float number to transform.
        '''
        if self.format == np.float16:
            [d] = struct.unpack(">H", struct.pack(">e", f))
            return f'{d:016b}'
        elif self.format == np.float32:
            [d] = struct.unpack(">I", struct.pack(">f", f))
            return f'{d:032b}'
        elif self.format == np.float64:
            [d] = struct.unpack(">Q", struct.pack(">d", f))
            return f'{d:064b}'

    def EqualsFloat(self, actualFloat):
        assert type(actualFloat) == self.format
        mine = self.S + self.E + self.T
        target = self.float2bin(actualFloat)
        return mine == target

    def PopulateData(self):
        if self.format == np.float16:
            self.CommonName = 'Half Precision'
            self.Base = 2
            self.SignificandBits = 10
            self.DecimalDigits = 3.31
            self.ExponentBits = 5
            self.DecimalEMax = 4.51
            self.ExponentBias = 2**4-1
            self.EMin = -14
            self.EMax = 15
            bitsSoFar = 0
            self.S = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+1)]
            bitsSoFar += 1
            self.E = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+self.ExponentBits)]
            bitsSoFar += self.ExponentBits
            self.T = self.float2bin(self.original)[bitsSoFar:(bitsSoFar + self.SignificandBits)]
            assert bitsSoFar + self.SignificandBits == 16
        elif self.format == np.float32:
            self.CommonName = 'Single Precision'
            self.Base = 2
            self.SignificandBits = 23
            self.DecimalDigits = 7.22
            self.ExponentBits = 8
            self.DecimalEMax = 38.23
            self.ExponentBias = 2**7-1
            self.EMin = -126
            self.EMax = 127
            bitsSoFar = 0
            self.S = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+1)]
            bitsSoFar += 1
            self.E = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+self.ExponentBits)]
            bitsSoFar += self.ExponentBits
            self.T = self.float2bin(self.original)[bitsSoFar:(bitsSoFar + self.SignificandBits)]
            assert bitsSoFar + self.SignificandBits == 32
        else:
            raise Exception('Unacceptable format')

    @staticmethod
    def Add(a, b, manual=False):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original + b.original)
        inexact = False
        lostBits = ''
        pInf = (0, (1 << a.ExponentBits) - 1, 0)
        nInf = (1, (1 << a.ExponentBits) - 1, 0)
        NaN = (1, (1 << a.ExponentBits) - 1, 1 << (a.SignificandBits - 1))

        c = MyFloat(valueFormat=a.format)
        AS, AE, AT = [int(_, 2) for _ in [a.S, a.E, a.T]]
        BS, BE, BT = [int(_, 2) for _ in [b.S, b.E, b.T]]
        CS, CE, CT = [int(_, 2) for _ in [c.S, c.E, c.T]]
        # Make sure A has the larger magnitude
        if AE < BE or (AE == BE and AT < BT):
            tmpS, tmpE, tmpT = AS, AE, AT
            AS, AE, AT = BS, BE, BT
            BS, BE, BT = tmpS, tmpE, tmpT
        # Detect special cases
        aInf = MyFloat.IsInfinite(AE, AT, a.ExponentBits)
        bInf = MyFloat.IsInfinite(BE, BT, b.ExponentBits)
        aNaN = MyFloat.IsNaN(AE, AT, a.ExponentBits)
        bNaN = MyFloat.IsNaN(BE, BT, b.ExponentBits)
        # Cannot add +inf to -inf
        if aInf and bInf:
            if AS != BS:
                CS, CE, CT = NaN
            else:
                CS, CE, CT = pInf if AS == 0 else nInf
        elif aInf ^ bInf:
            CS, CE, CT = pInf
            CS = AS if aInf else BS
        elif aNaN or bNaN:
            CS, CE, CT = (AS, AE, AT) if aNaN else (BS, BE, BT)
        else:
            # Normal operation
            # Set the exponent
            CE = AE

            # If a >> b, all trailing bits from b are lost so replace them with a's trailing bits
            # TODO check this math
            if AE - BE >= a.SignificandBits:
                CT = AT
                CS = AS
            else:
                # Add in implicit leading digit
                leadingDigitA = 0 if AE == 0 else 1
                leadingDigitB = 0 if BE == 0 else 1
                AT |= leadingDigitA << a.SignificandBits
                BT |= leadingDigitB << b.SignificandBits

                # Scale right
                shiftAmt = AE - BE
                lostBits = bin(BT % (1 << shiftAmt))[2:]
                BT >>= shiftAmt

                # Add
                if AS != BS:
                    CS = AS if AT > BT else BS
                    CT = abs(AT - BT)
                else:
                    CT = AT + BT
                    CS = AS

            # Normalize
            CE, CT, _, lostBits = MyFloat.Normalize(CE, CT, c.ExponentBits, c.SignificandBits, lostBits)

        c.S = format(CS, '01b')
        c.E = format(CE, '0' + str(a.ExponentBits) + 'b')
        c.T = format(CT, '0' + str(a.SignificandBits) + 'b')
        # We probably messed up normalization if these assertions fail
        assert len(c.S) == 1
        assert len(c.E) == a.ExponentBits
        assert len(c.T) == a.SignificandBits
        c.Reinterpret()
        return c

    @staticmethod
    def IsInfinite(exponent, significand, exponentbits):
        return exponent == (1 << exponentbits) - 1 and significand == 0

    @staticmethod
    def IsNaN(exponent, significand, exponentbits):
        return exponent == (1 << exponentbits) - 1 and significand != 0

    @staticmethod
    def Normalize(exponent, significand, exponentBits, significandBits, lostBits):
        infty = False
        infoLost = False
        if significand >= 1 << (significandBits + 1):
            # Significand >= 2
            # Rightshift significand and add to exponent until infinity
            # or 1 <= significand < 2
            rShifted = 0
            while significand >= 1 << (significandBits + 1):
                bitToLose = (significand % 2) << rShifted
                lostBits = str(bitToLose) + lostBits
                if bitToLose == 1:
                    infoLost = True
                significand >>= 1
                exponent += 1
                # Check for infinity
                if exponent == (1 << exponentBits) - 1:
                    significand = 0
                    break
            else:
                # Clear out the implicit leading 1 in the significand
                significand &= (1 << significandBits) - 1
            if infoLost:
                print('lost bits: ', lostBits)

        elif exponent > 0:
            # 0 <= significand < 2
            # Leftshift significand and subtract from exponent until subnormal
            # or 1 <= significand < 2
            while significand < 1 << significandBits:
                significand <<= 1
                exponent -= 1
                # Check for subnormal
                if exponent == 0:
                    break
            else:
                # Clear out the implicit leading 1 in the significand
                significand &= (1 << significandBits) - 1
            pass
        elif exponent == 0 and significand >= 1 << significandBits:
            # subnormal getting upgraded
            exponent = 1
            significand &= (1 << significandBits) - 1
            # while significand >= 1 << significandBits:
        elif exponent == 0:
            # once a denormal always a denormal!
            pass


        else:
            raise Exception('Unexpected case')

        return exponent, significand, infoLost, lostBits


    @staticmethod
    def Multiply(a, b, manual=False):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original * b.original)
        else:
            raise NotImplementedError()


def TestAdd():
    testCases = [
        [1, -2],
        [1.001, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, -0.0],
        [-0.0, 0.0],
        [-0.0, -0.0],
        [np.infty - np.infty, np.nan],
        [np.nan, np.infty - np.infty],
        [1.0/3.0, 2.0/3.0],
        [3, 7],
        [30, 70],
        [300, 700],
        [3000, 7000],
        [3, 7000],
        [-27, np.infty],
        [np.infty, np.infty],
        [np.infty, -np.infty],
        [-np.infty, -np.infty],
        [-np.infty, np.infty],
        [np.NaN, 0],
        [3.052E-5, 3.052E-5], #subnormal + subnormal = normal
        [6.104E-5, -3.052E-5], #normal + subnormal = subnormal
        [6.104E-5, -6.11E-5], #normal + normal = subnormal

    ]
    for a, b in testCases:
        A = MyFloat(a, np.float16)
        B = MyFloat(b, np.float16)
        C = MyFloat.Add(A, B, True)
        mine = C.original
        actual = A.original + B.original
        if C.EqualsFloat(actual):
            print('----------------')
            print(A, '+', B)
            print('Matches: ', MyFloat(actual))
        else:
            print('----------------')
            print(A, '+', B)
            print('FAILED: ', C, ' Actual: ', MyFloat(actual))

def Test():
    print(MyFloat(np.float16(np.inf)))
    print(MyFloat(np.float16(np.inf-np.inf)))
    print(MyFloat(np.float16(np.NaN)))
    print(MyFloat(np.float16(0)))
    print(MyFloat(np.float16(1)))
    print(MyFloat(np.float16(2**14)))
    print(MyFloat(np.float16(2**15)))
    print(MyFloat(np.float16(2**16)))


if __name__ == '__main__':
    TestAdd()