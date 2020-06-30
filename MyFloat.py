import numpy as np
import math
import random as rand
import struct
from enum import Enum
import operator
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

    # https://www.research.ibm.com/haifa/projects/verification/fpgen/papers/ieee-test-suite-v2.pdf
    # Binary floating-point types
    # Zero
    @staticmethod
    def _Zero(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' * retVal.ExponentBits
        retVal.T = '0' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def Zero(self):
        return MyFloat._Zero(self.format)

    # Smallest possible subnormal number
    @staticmethod
    def _MinSubNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' * retVal.ExponentBits
        retVal.T = ('0' * (retVal.SignificandBits - 1)) + '1'
        retVal.Reinterpret()
        return retVal
    def MinSubNorm(self):
        return MyFloat._MinSubNorm(self.format)

    # Smallest number larger than the smallest possible subnormal number
    @staticmethod
    def _NextMinSubNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' * retVal.ExponentBits
        retVal.T = ('0' * (retVal.SignificandBits - 2)) + '10'
        retVal.Reinterpret()
        return retVal
    def NextMinSubNorm(self):
        return MyFloat._NextMinSubNorm(self.format)

    # Middle subnormal number in total ordering of the subnormals
    @staticmethod
    def _MidSubNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        minSub = MyFloat._MinSubNorm(valueFormat)
        maxSub = MyFloat._MaxSubNorm(valueFormat)
        retVal.T = format((int(minSub.T, 2) + int(maxSub.T, 2)) // 2, '0' + str(retVal.SignificandBits) + 'b')
        retVal.Reinterpret()
        return retVal
    def MidSubNorm(self):
        return MyFloat._MidSubNorm(self.format)

    # Largest number smaller than the largest possible subnormal number
    @staticmethod
    def _PrevMaxSubNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' * retVal.ExponentBits
        retVal.T = ('1' * (retVal.SignificandBits - 1)) + '0'
        retVal.Reinterpret()
        return retVal
    def PrevMaxSubNorm(self):
        return MyFloat._PrevMaxSubNorm(self.format)

    # Largest possible subnormal number
    @staticmethod
    def _MaxSubNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' * retVal.ExponentBits
        retVal.T = '1' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def MaxSubNorm(self):
        return MyFloat._MaxSubNorm(self.format)

    # Smallest possible normal number
    @staticmethod
    def _MinNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = ('0' * (retVal.ExponentBits - 1)) + '1'
        retVal.T = '0' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def MinNorm(self):
        return MyFloat._MinNorm(self.format)

    # Smallest number larger than the smallest possible normal number
    @staticmethod
    def _NextMinNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = ('0' * (retVal.ExponentBits - 1)) + '1'
        retVal.T = ('0' * (retVal.SignificandBits - 1)) + '1'
        retVal.Reinterpret()
        return retVal
    def NextMinNorm(self):
        return MyFloat._NextMinNorm(self.format)

    # Middle normal number in total ordering of the normals
    @staticmethod
    def _MidNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        minNorm = MyFloat._MinNorm(valueFormat)
        maxNorm = MyFloat._MaxNorm(valueFormat)
        retVal.E = format((int(minNorm.E, 2) + int(maxNorm.E, 2)) // 2, '0' + str(retVal.ExponentBits) + 'b')
        retVal.T = format((int(minNorm.T, 2) + int(maxNorm.T, 2)) // 2, '0' + str(retVal.SignificandBits) + 'b')
        retVal.Reinterpret()
        return retVal
    def MidNorm(self):
        return MyFloat._MidNorm(self.format)

    # Largest number smaller than the largest possible normal number
    @staticmethod
    def _PrevMaxNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = ('1' * (retVal.ExponentBits - 1)) + '0'
        retVal.T = ('1' * (retVal.SignificandBits - 1)) + '0'
        retVal.Reinterpret()
        return retVal
    def PrevMaxNorm(self):
        return MyFloat._PrevMaxNorm(self.format)

    # Largest possible normal number
    @staticmethod
    def _MaxNorm(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = ('1' * (retVal.ExponentBits - 1)) + '0'
        retVal.T = '1' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def MaxNorm(self):
        return MyFloat._MaxNorm(self.format)

    # Positive infinity
    @staticmethod
    def _Infinity(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '1' * retVal.ExponentBits
        retVal.T = '0' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def Infinity(self):
        return MyFloat._Infinity(self.format)

    # Default NaN
    @staticmethod
    def _DefaultNaN(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '1' * retVal.ExponentBits
        retVal.T = '1' + ('0' * (retVal.SignificandBits - 1))
        retVal.Reinterpret()
        return retVal
    def DefaultNaN(self):
        return MyFloat._DefaultNaN(self.format)

    # Largest number smaller than one
    @staticmethod
    def _PrevOne(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' + ('1' * (retVal.ExponentBits - 2)) + '0'
        retVal.T = '1' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def PrevOne(self):
        return MyFloat._PrevOne(self.format)

    # One
    @staticmethod
    def _One(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' + ('1' * (retVal.ExponentBits - 1))
        retVal.T = '0' * retVal.SignificandBits
        retVal.Reinterpret()
        return retVal
    def One(self):
        return MyFloat._One(self.format)

    # Smallest number larger than one
    @staticmethod
    def _NextOne(valueFormat):
        retVal = MyFloat(valueFormat=valueFormat)
        retVal.S = '0'
        retVal.E = '0' + ('1' * (retVal.ExponentBits - 1))
        retVal.T = ('0' * (retVal.SignificandBits - 1)) + '1'
        retVal.Reinterpret()
        return retVal
    def NextOne(self):
        return MyFloat._NextOne(self.format)

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
    def MultSign(S1, S2):
        return 0 if S1 == S2 else 1

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
        # Infinity
        elif aInf or bInf:
            # Cannot multiply infinity by zero
            if aZero or bZero:
                CS, CE, CT = NaN
            else:
                # otherwise return infinity with the proper sign
                CS, CE, CT = pInf if MyFloat.MultSign(AS, BS) == 0 else nInf
        elif aZero or bZero:
            CS = MyFloat.MultSign(AS, BS)
            CE, CT = (AE, AT) if aZero else (BE, BT)
        else:
            # Normal operation

            # Add in implicit leading digit
            leadingDigitA = 0 if AE == 0 else 1
            leadingDigitB = 0 if BE == 0 else 1
            AT |= leadingDigitA << a.SignificandBits
            BT |= leadingDigitB << b.SignificandBits

            roundingBits = Rounding()
            # Mult
            CS = MyFloat.MultSign(AS, BS)
            CE = max(AE, 1) + max(BE, 1) - c.ExponentBias
            # At this point CE is the actual unbiased exponent
            if CE <= 0:
                CE -= 1
            CT = AT * BT
            CT = roundingBits.RShift(CT, c.SignificandBits)

            # Normalize
            CE, CT, roundingBits = MyFloat.Normalize(CE, CT, c.ExponentBits, c.SignificandBits, roundingBits)

            CE, CT = MyFloat.Round(CE, CT, c.ExponentBits, c.SignificandBits, roundingBits)

        c.S = format(CS, '01b')
        c.E = format(CE, '0' + str(a.ExponentBits) + 'b')
        c.T = format(CT, '0' + str(a.SignificandBits) + 'b')
        # We probably messed up normalization if these assertions fail
        assert len(c.S) == 1
        assert len(c.E) == a.ExponentBits
        assert len(c.T) == a.SignificandBits
        c.Reinterpret()
        if MyFloat.logFailures and MyFloat(a.original * b.original).original != c.original:
            MyFloat.failureDict[(a.original, b.original)] = (MyFloat(a.original * b.original).original, c.original)
        return c

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
            shiftAmt = max(AE, 1) - max(BE, 1)
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
            CE, CT, roundingBits = MyFloat.Normalize(CE, CT, c.ExponentBits, c.SignificandBits, roundingBits)

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
        # Significand >= 2
        # Rightshift significand and add to exponent until infinity
        # or 1 <= significand < 2
        while significand >= 1 << (significandBits + 1):
            # Check for infinity
            if exponent >= (1 << exponentBits) - 1:
                break
            significand = roundingBits.RShift(significand, 1)
            exponent += 1

        # 0 <= significand < 2
        # Leftshift significand and subtract from exponent until subnormal
        # or 1 <= significand < 2
        while significand < 1 << significandBits:
            # Check for subnormal
            if exponent <= 0:
                break
            if exponent != 1:
                # check if it's okay to LShift
                significand = roundingBits.LShift(significand, 1)
            exponent -= 1

        # Very subnormal number...
        if exponent < 0:
            significand = roundingBits.RShift(significand, abs(exponent))
            exponent = 0

        # Overflow
        if exponent >= (1 << exponentBits) - 1:
            exponent = (1 << exponentBits) - 1
            significand = 0
            roundingBits.Clear()

        # special case of subnormal upgrading to normal
        if exponent == 0 and significand >= (1 << significandBits):
            exponent = 1

        # Clear out the implicit leading 1 in the significand
        significand &= (1 << significandBits) - 1
        return exponent, significand, roundingBits

    @staticmethod
    def OldNormalize(exponent, significand, exponentBits, significandBits, roundingBits):
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
        elif exponent <= 0 and significand >= 1 << significandBits:
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

