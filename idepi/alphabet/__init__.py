
from __future__ import division, print_function

from collections import defaultdict
from copy import deepcopy

from numpy import dtype


__all__ = ['Alphabet']


class ConstantFactory:
    __slots__ = ('__value',)
    def __init__(self, value):
        self.__value = value
    def __call__(self):
        return self.__value


class Alphabet:
    AMINO, DNA, STANFEL, CUSTOM = 0, 1, 2, 3

    # underscore must go first or re.compile blows up
    GAPS = '_.-='
    GAP_REPR = '[]'

    __DNA_ALPH   = 'ACGTUN-'
    __AMINO_ALPH = 'ACGILMPSTVDENQFWYHKRX-'
    __STANFEL_ALPH = {
        'A': 0, 'C': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'P': 0, 'S': 0, 'T': 0, 'V': 0,
        'D': 1, 'E': 1, 'N': 1, 'Q': 1,
        'F': 2, 'W': 2, 'Y': 2,
        'H': 3, 'K': 3, 'R': 3,
        'X': 4,
        '-': 5 # this is a space, but we cant have 'keyword be an expression'
    }

    def __init__(self, mode=None, chars=None):

        if mode is None:
           mode = self.AMINO

        if mode == Alphabet.STANFEL:
            d = Alphabet.__STANFEL_ALPH
            l = []
            # don't forget to +1 here, otherwise we miss the '-' character
            for i in range(max(d.values()) + 1):
                l.append('[%s]' % ''.join(sorted(k if k not in Alphabet.GAPS else '' for k, v_ in d.items() if i == v_)))

        elif mode in (Alphabet.AMINO, Alphabet.DNA):
            alph = Alphabet.__AMINO_ALPH if mode == self.AMINO else Alphabet.__DNA_ALPH
            d, l = Alphabet.__dict_and_list(alph)

        elif mode == Alphabet.CUSTOM:
            if chars is None:
                raise ValueError('Custom Alphabet requires a `chars\' parameter')
            d, l = Alphabet.__dict_and_list(chars)

        else:
            raise ValueError('mode must be one of Alphabet.AMINO, Alphabet.DNA, Alphabet.STANFEL, or Alphabet.CUSTOM')

        default = 'N' if mode == Alphabet.DNA else 'X'

        self.__dict = defaultdict(ConstantFactory(d[default]))
        self.__dict.update(d)
        self.__list = l

    def __len__(self):
        return len(self.__list)

    def __call__(self, char):
        if not isinstance(char, str):
            raise ValueError('alphabet() converts chars to alphabet indices')
        if char in Alphabet.GAPS:
            char = '-'
        return self.__dict[char]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('alphabet[] converst alpahbet indices to a str repr')
        return self.__list[idx]

    @staticmethod
    def __dict_and_list(alphabet):
        alph = alphabet.upper()
        d = dict((alph[i], i) for i in range(len(alph)))
        l = [alph[i] if alph[i] not in Alphabet.GAPS else Alphabet.GAP_REPR for i in range(len(alph))]
        return d, l

    def todict(self):
        # return a manual deepcopy, as deepcopy is currently barfing
        d = defaultdict(ConstantFactory(self.__dict['X']))
        d.update(self.__dict)
        return d

    def tolist(self):
        return deepcopy(self.__list)

    def todtype(self):
        return dtype(('b1', (len(self.__list),)))
