import numpy as np
import math
import random as rand
import struct
from enum import Enum
import operator
# http://weitz.de/ieee/
# https://ieeexplore.ieee.org/abstract/document/4380621/authors#authors

# I am intentionally hitting failure cases to make sure I handle them properly
np.seterr(all='ignore')
op2str = {
    operator.add: '+',
    operator.sub: '-',
    operator.mul: '*',
}
randomizedCases = 10_000

class FloatFlag(Enum):
    OK = 0
    INVALID = 1 << 0
    DIV0 = 1 << 1
    OVERFLOW = 1 << 2
    UNDERFLOW = 1 << 3
    INEXACT = 1 << 4

class MyFloat:
    status = FloatFlag.OK
    logFailures = True
    failureDict = dict()
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
    def __sub__(self, other):
        return MyFloat.Sub(self, other)
    def __rsub__(self, other):
        return MyFloat.Sub(self, other)
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

    def bin2float(self, b):
        return MyFloat.bin2floatStatic(self.format, b)

    # https://stackoverflow.com/a/59594903
    @staticmethod
    def bin2floatStatic(myFormat, b):
        ''' Convert binary string to a float.

        Attributes:
            :b: Binary string to transform.
        '''
        if myFormat == np.float16:
            h = int(b, 2).to_bytes(2, byteorder="big")
            return np.float16(struct.unpack('>e', h)[0])
        elif myFormat == np.float32:
            h = int(b, 2).to_bytes(4, byteorder="big")
            return np.float32(struct.unpack('>f', h)[0])
        elif myFormat == np.float64:
            h = int(b, 2).to_bytes(8, byteorder="big")
            return np.float64(struct.unpack('>d', h)[0])


    def float2bin(self, b):
        return MyFloat.float2binStatic(self.format, b)

    # https://stackoverflow.com/a/59594903
    @staticmethod
    def float2binStatic(myFormat, f):
        ''' Convert float to n-bit binary string.

        Attributes:
            :f: Float number to transform.
        '''
        if myFormat == np.float16:
            [d] = struct.unpack(">H", struct.pack(">e", f))
            return f'{d:016b}'
        elif myFormat == np.float32:
            [d] = struct.unpack(">I", struct.pack(">f", f))
            return f'{d:032b}'
        elif myFormat == np.float64:
            [d] = struct.unpack(">Q", struct.pack(">d", f))
            return f'{d:064b}'

    def EqualsFloatBits(self, actualFloat):
        assert type(actualFloat) == self.format
        # not being nitpicky as long as they're both NaN's
        if math.isnan(self.original) or math.isnan(actualFloat):
            return math.isnan(self.original) and math.isnan(actualFloat)
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
    def Multiply(a, b, manual=True):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original * b.original)
        pInf = (0, (1 << a.ExponentBits) - 1, 0)
        nInf = (1, (1 << a.ExponentBits) - 1, 0)
        NaN = (1, (1 << a.ExponentBits) - 1, 1 << (a.SignificandBits - 1))
        pZero = (0, 0, 0)
        nZero = (1, 0, 0)

        c = MyFloat(valueFormat=a.format)
        AS, AE, AT = [int(_, 2) for _ in [a.S, a.E, a.T]]
        BS, BE, BT = [int(_, 2) for _ in [b.S, b.E, b.T]]
        CS, CE, CT = [int(_, 2) for _ in [c.S, c.E, c.T]]

        aInf = MyFloat._IsInfinite(AE, AT, a.ExponentBits)
        bInf = MyFloat._IsInfinite(BE, BT, b.ExponentBits)
        aNaN = MyFloat._IsNaN(AE, AT, a.ExponentBits)
        bNaN = MyFloat._IsNaN(BE, BT, b.ExponentBits)
        aZero = MyFloat._IsZero(AE, AT)
        bZero = MyFloat._IsZero(BE, BT)

        # Detect special cases
        # NaN
        if aNaN or bNaN:
            CS, CE, CT = (AS, AE, AT) if aNaN else (BS, BE, BT)
        # Cannot add +inf to -inf
        elif aInf and bInf:
            if AS != BS:
                CS, CE, CT = NaN
            else:
                CS, CE, CT = pInf if AS == 0 else nInf
        # Adding infinite to finite
        elif aInf ^ bInf:
            CS, CE, CT = pInf
            CS = AS if aInf else BS
        # +-Zero plus +-Zero
        elif aZero and bZero:
            if AS == 1 and BS == 1:
                CS, CE, CT = nZero
            else:
                CS, CE, CT = pZero
        # x + -x = +0 always
        elif AS != BS and AE == BE and AT == BT:
            CS, CE, CT = pZero
        # x + +-0 = x for x different from 0
        elif aZero or bZero:
            CS, CE, CT = (AS, AE, AT) if bZero else (BS, BE, BT)
        else:
            pass

    @staticmethod
    def Add(a, b, manual=True):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original + b.original)
        pInf = (0, (1 << a.ExponentBits) - 1, 0)
        nInf = (1, (1 << a.ExponentBits) - 1, 0)
        NaN = (1, (1 << a.ExponentBits) - 1, 1 << (a.SignificandBits - 1))
        pZero = (0, 0, 0)
        nZero = (1, 0, 0)

        c = MyFloat(valueFormat=a.format)
        AS, AE, AT = [int(_, 2) for _ in [a.S, a.E, a.T]]
        BS, BE, BT = [int(_, 2) for _ in [b.S, b.E, b.T]]
        CS, CE, CT = [int(_, 2) for _ in [c.S, c.E, c.T]]
        # Make sure A has the larger magnitude
        if AE < BE or (AE == BE and AT < BT):
            tmpS, tmpE, tmpT = AS, AE, AT
            AS, AE, AT = BS, BE, BT
            BS, BE, BT = tmpS, tmpE, tmpT

        aInf = MyFloat._IsInfinite(AE, AT, a.ExponentBits)
        bInf = MyFloat._IsInfinite(BE, BT, b.ExponentBits)
        aNaN = MyFloat._IsNaN(AE, AT, a.ExponentBits)
        bNaN = MyFloat._IsNaN(BE, BT, b.ExponentBits)
        aZero = MyFloat._IsZero(AE, AT)
        bZero = MyFloat._IsZero(BE, BT)

        # Detect special cases
        # NaN
        if aNaN or bNaN:
            CS, CE, CT = (AS, AE, AT) if aNaN else (BS, BE, BT)
        # Cannot add +inf to -inf
        elif aInf and bInf:
            if AS != BS:
                CS, CE, CT = NaN
            else:
                CS, CE, CT = pInf if AS == 0 else nInf
        # Adding infinite to finite
        elif aInf ^ bInf:
            CS, CE, CT = pInf
            CS = AS if aInf else BS
        # +-Zero plus +-Zero
        elif aZero and bZero:
            if AS == 1 and BS == 1:
                CS, CE, CT = nZero
            else:
                CS, CE, CT = pZero
        # x + -x = +0 always
        elif AS != BS and AE == BE and AT == BT:
            CS, CE, CT = pZero
        # x + +-0 = x for x different from 0
        elif aZero or bZero:
            CS, CE, CT = (AS, AE, AT) if bZero else (BS, BE, BT)
        else:
            # Normal operation
            # Set the exponent
            CE = AE


            # Add in implicit leading digit
            leadingDigitA = 0 if AE == 0 else 1
            leadingDigitB = 0 if BE == 0 else 1
            AT |= leadingDigitA << a.SignificandBits
            BT |= leadingDigitB << b.SignificandBits

            # Scale right
            # when shifting into subnormals, the significand weighting is the same for exponent 0 and 1
            shiftAmt = max(AE - max(BE, 1), 0)
            roundingBits = Rounding()
            BT = roundingBits.RShift(BT, shiftAmt)
            BE -= shiftAmt
            # Add
            if AS != BS:
                CS = AS if AT > BT else BS
                # subtraction here must include lost bits due to rounding
                CT = roundingBits.Subtract(AT, BT)
            else:
                CT = AT + BT
                CS = AS

            # Normalize
            CE, CT, _, roundingBits = MyFloat.Normalize(CE, CT, c.ExponentBits, c.SignificandBits, roundingBits)

            CE, CT = MyFloat.Round(CE, CT, c.ExponentBits, c.SignificandBits, roundingBits)

        c.S = format(CS, '01b')
        c.E = format(CE, '0' + str(a.ExponentBits) + 'b')
        c.T = format(CT, '0' + str(a.SignificandBits) + 'b')
        # We probably messed up normalization if these assertions fail
        assert len(c.S) == 1
        assert len(c.E) == a.ExponentBits
        assert len(c.T) == a.SignificandBits
        c.Reinterpret()
        if MyFloat.logFailures and MyFloat(a.original + b.original).original != c.original:
            MyFloat.failureDict[(a.original, b.original)] = (MyFloat(a.original + b.original).original, c.original)
        return c

    @staticmethod
    def Sub(a, b, manual=True):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original - b.original)
        if a.IsNaN() or b.IsNaN():
            return MyFloat.Add(a, b, manual=True)
        bNeg = MyFloat(b.original)
        bNeg.S = '0' if bNeg.S == '1' else '1'
        bNeg.Reinterpret()
        return MyFloat.Add(a, bNeg)

    @staticmethod
    def rShiftSignificand(significand, shiftAmt, prevRoundingBits):
        prevGuardBit, prevRoundBit, prevStickyBit = prevRoundingBits
        guardBit, roundBit, stickyBit = prevRoundingBits
        lostBits = format(significand % (1 << shiftAmt), '0' + str(shiftAmt) + 'b')
        if shiftAmt == 0:
            pass
        elif shiftAmt == 1:
            guardBit = lostBits[0] == '1'
            roundBit = prevGuardBit
            stickyBit = prevRoundBit or prevStickyBit
        elif shiftAmt == 2:
            guardBit = lostBits[0] == '1'
            roundBit = lostBits[1] == '1'
            stickyBit = prevGuardBit or prevRoundBit or prevStickyBit
        else:
            guardBit = lostBits[0] == '1'
            roundBit = lostBits[1] == '1'
            stickyBit = ('1' in lostBits[2:]) or prevGuardBit or prevRoundBit or prevStickyBit

        return (guardBit, roundBit, stickyBit)

    def IsInfinite(self):
        return MyFloat._IsInfinite(self.E, self.T, self.ExponentBits)

    @staticmethod
    def _IsInfinite(exponent, significand, exponentbits):
        return exponent == (1 << exponentbits) - 1 and significand == 0

    def IsNaN(self):
        return MyFloat._IsNaN(self.E, self.T, self.ExponentBits)

    @staticmethod
    def _IsNaN(exponent, significand, exponentbits):
        return exponent == (1 << exponentbits) - 1 and significand != 0

    def IsZero(self):
        return MyFloat._IsZero(self.E, self.T)

    @staticmethod
    def _IsZero(exponent, significand):
        return exponent == 0 and significand == 0

    @staticmethod
    def Normalize(exponent, significand, exponentBits, significandBits, roundingBits):
        infty = False
        infoLost = False
        if significand >= 1 << (significandBits + 1):
            # Significand >= 2
            # Rightshift significand and add to exponent until infinity
            # or 1 <= significand < 2
            rShifted = 0
            while significand >= 1 << (significandBits + 1):
                significand = roundingBits.RShift(significand, 1)
                exponent += 1
                # Check for infinity
                if exponent == (1 << exponentBits) - 1:
                    significand = 0
                    roundingBits.Clear()
                    break
            else:
                # Clear out the implicit leading 1 in the significand
                significand &= (1 << significandBits) - 1
            if infoLost:
                pass
                # print('lost bits: ', lostBits)

        elif exponent > 0:
            # 0 <= significand < 2
            # Leftshift significand and subtract from exponent until subnormal
            # or 1 <= significand < 2
            while significand < 1 << significandBits:
                if exponent != 1:
                    # check if it's okay to LShift
                    significand = roundingBits.LShift(significand, 1)
                exponent -= 1
                # Check for subnormal
                if exponent == 0:
                    break
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

        return exponent, significand, infoLost, roundingBits

    # https://stackoverflow.com/questions/19146131/rounding-floating-point-numbers-after-addition-guard-sticky-and-round-bits#:~:text=The%20Guard%20bit%20is%20the,no%20other%20bit%20is%20present.
    # http://pages.cs.wisc.edu/~david/courses/cs552/S12/handouts/guardbits.pdf
    # round to nearest, ties to even
    @staticmethod
    def Round(exponent, significand, exponentBits, significandBits, roundingBits):
        # slicing operator returns an empty list if range isn't applicable to lostBits
        assert isinstance(roundingBits, Rounding)
        guardBit, roundBit, stickyBit = (roundingBits.guard, roundingBits.round, roundingBits.sticky)

        # round down
        # ...0xx
        if not guardBit:
            pass
        # round up
        # ...1xx where at least one x is one
        elif guardBit and (roundBit or stickyBit):
            significand += 1
        # round even
        # ...100
        elif guardBit and not roundBit and not stickyBit:
            if significand % 2 == 1:
                significand += 1

        # check for rounding overflow
        if significand == (1 << significandBits):
            exponent += 1
            significand = 0
        return exponent, significand

class Rounding:
    def __init__(self):
        self.lostBits = ''

    def __str__(self):
        return "'" + self.lostBits + "': " + str([self.guard, self.round, self.sticky])
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return str(self)

    @property
    def guard(self):
        return '1' in self.lostBits[0:1]

    @property
    def round(self):
        return '1' in self.lostBits[1:2]

    @property
    def sticky(self):
        return '1' in self.lostBits[2:]

    def Clear(self):
        self.lostBits = ''

    def RShift(self, val, shiftAmt):
        if shiftAmt == 0:
            return val
        self.lostBits = format(val % (1 << shiftAmt), '0' + str(shiftAmt) + 'b') + self.lostBits
        return val >> shiftAmt

    def LShift(self, val, shiftAmt):
        if len(self.lostBits) >= shiftAmt:
            retval = (val << shiftAmt) | (int(self.lostBits, 2) >> (len(self.lostBits) - shiftAmt))
            self.lostBits = self.lostBits[shiftAmt:]
        elif self.lostBits == '':
            retval = val << shiftAmt
        else:
            retval = (val << shiftAmt) | (int(self.lostBits, 2) << (shiftAmt - len(self.lostBits)))
            self.Clear()
        return retval

    def Subtract(self, valA, valB):
        # I need to track bits that were right-shifted past the significand and account for them here.
        shiftAmt = len(self.lostBits)
        if shiftAmt == 0:
            return valA - valB
        valA <<= shiftAmt
        valB <<= shiftAmt
        valB |= int(self.lostBits, 2)
        self.lostBits = ''
        return self.RShift(valA - valB, shiftAmt)

def TestOp(operation):
    currType = np.float16
    manualCases = [
        [0.5874, -0.1327],
        [2050, 3],
        [0.7217, 1.865E-3],
        [1.0/3.0, 2.0/3.0],
        [2048, 3],
        [1, 0],
        [1, -2],
        [1, -1],
        [1.001, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, -0.0],
        [-0.0, 0.0],
        [-0.0, -0.0],
        [np.infty - np.infty, np.nan], # NaN from invalid calc + plain NaN
        [np.nan, np.infty - np.infty], # plain NaN + NaN from invalid calc
        [3, 7],
        [30, 70],
        [300, 700],
        [3000, 7000],
        [3, 7000],
        [1, 2047],
        [2, 2047],
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
    utahCases = []
    randomCases = []
    previouslyFailedRandomCases = []
    with open('UtahActualCalculationErrors.txt') as f:
        utahCases.extend([[currType(x) for x in line.split(' ')] for line in f.readlines()])
    for _ in range(randomizedCases):
        randomCases.append(
            [
                MyFloat.bin2floatStatic(\
                    currType,
                    format(
                        rand.getrandbits(np.dtype(currType).itemsize * 8),
                        '01b')),
                MyFloat.bin2floatStatic(\
                    currType,
                    format(
                        rand.getrandbits(np.dtype(currType).itemsize * 8),
                        '01b'))
            ])
    with open('RandomTestFailures.txt') as f:
        previouslyFailedRandomCases.extend([[currType(x) for x in line.split(' ')] for line in f.readlines()])

    allCases = []
    allCases.extend(manualCases)
    allCases.extend(utahCases)
    allCases.extend(randomCases)
    allCases.extend(previouslyFailedRandomCases)
    passed = 0
    total = 0
    for a, b in allCases:
        A = MyFloat(a, currType)
        B = MyFloat(b, currType)
        C = operation(A, B)
        mine = C.original
        actual = operation(A.original, B.original)
        if C.EqualsFloatBits(actual):
            passed += 1
            # print('----------------')
            # print(A, op2str[operation], B)
            # print('Matches: ', MyFloat(actual))
            pass
        else:
            print('----------------')
            print(A, op2str[operation], B)
            # print(A.original, B.original)
            print('FAILED: ', C, ' Actual: ', MyFloat(actual))
        total += 1
    print(op2str[operation] + ':', str(100*passed/total) + '%')



if __name__ == '__main__':
    # TestOp(operator.add)
    # TestOp(operator.sub)
    TestOp(operator.mul)