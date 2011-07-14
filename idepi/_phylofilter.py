
import json
from math import floor
from os import close, remove
from os.path import dirname, exists, join, realpath
from tempfile import mkstemp

import numpy as np

from Bio import SeqIO

from _alphabet import Alphabet
from _basefilter import BaseFilter
from _hyphy import HyPhy
from _util import is_HXB2


__all__ = ['PhyloFilter']


class PhyloFilter(BaseFilter):

    def __init__(self, alphabet=None, batchfile=None, ref_id_func=None, skip_func=None):
        if batchfile is None:
            batchfile = join(dirname(realpath(__file__)), '..', 'res', 'CorrectForPhylogeny.bf')

        if not exists(batchfile):
            raise ValueError('Please pass a valid (and existing) batchfile to PhyloFilter()')

        if alphabet is None:
            alphabet = Alphabet()
        if ref_id_func is None:
            ref_id_func = is_HXB2
        if skip_func is None:
            skip_func = lambda x: False

        fd, self.__inputfile = mkstemp(); close(fd)

        self.__alph = alphabet
        self.__batchfile = batchfile
        self.__rfn = ref_id_func
        self.__sfn = skip_func
#         self.__run, self.__data, self.__colnames = False, None, None

    def __del__(self):
        for file in (self.__inputfile,):
            if file and exists(file):
                remove(file)

    @staticmethod
    def __compute(alignment, alphabet, batchfile, inputfile, ref_id_func, skip_func, hyphy=None):
        if hyphy is None:
            hyphy = HyPhy()

        refseq = None
        for row in alignment:
            r = apply(ref_id_func, (row.id,))
            if r and refseq is None:
                refseq = row
            elif r:
                raise RuntimeError('Reference sequence found twice!?!?!?!')

        alignment = [row for row in alignment if not apply(ref_id_func, (row.id,)) and not apply(skip_func, (row.id,))]

        with open(inputfile, 'w') as fh:
            SeqIO.write(alignment, fh, 'fasta')

        HyPhy.execute(hyphy, batchfile, (inputfile,))

        _ids  = HyPhy.retrieve(hyphy, 'ids', HyPhy.MATRIX)
        mat  = HyPhy.retrieve(hyphy, 'data', HyPhy.MATRIX)
        order = HyPhy.retrieve(hyphy, 'order', HyPhy.STRING).strip(',').split(',')

        assert(_ids.mRows == 0)

        ids = [_ids.MatrixCell(0, i) for i in xrange(_ids.mCols)]

        ncol = mat.mCols / len(order) * len(alphabet)
        tmp = np.zeros((mat.mRows, ncol), dtype=float, order='F') # use column-wise order in memory

        # cache the result for each stride's indexing into the alphabet
        alphidx = [alphabet[order[i]] for i in xrange(len(order))]

        for i in xrange(mat.mRows):
            for j in xrange(mat.mCols):
                # we map j from HyPhy column order into self.__alph column order
                # by getting at the MSA column (j / len(order)), multiplying by
                # the self.__alph stride (len(self.__alph)), and then finally adding
                # the alphabet-specific index (alphidx[r])
                q = int(floor(j / len(order))) # quotient
                r = j % len(order) # remainder
                k = (q * len(alphabet)) + alphidx[r]
                tmp[i, k] += mat.MatrixCell(i, j)

        colsum = np.sum(tmp, axis=0)
        idxs = [i for i in xrange(ncol) if colsum[i] != 0.]
        ignore_idxs = set([i for i in xrange(ncol) if colsum[i] == 0.])

        data = tmp[:, idxs]

#         data = np.zeros((mat.mRows, ncol - len(ignore_idxs)), dtype=float)
# 
#         j = 0
#         for i in xrange(ncol):
#             if i in ignore_idxs:
#                 continue
#             data[:, j] = tmp[:, i]
#             j += 1

        if refseq is not None:
            alignment.append(refseq)

        colnames = BaseFilter._colnames(alignment, alphabet, ref_id_func, ignore_idxs)

        # make sure that the columns do line up
        assert(len(colnames) == data.shape[1])

        # return ids, mat, order, colnames
        return colnames, data

    def filter(self, alignment):
        return PhyloFilter.__compute(
            alignment, self.__alph, self.__batchfile, self.__inputfile, self.__rfn, self.__sfn
        )

#     @property
#     def data(self):
#         if not self.__run:
#             raise RuntimeError('No phylofiltering model computed')
#         return self.__data
# 
#     @property
#     def colnames(self):
#         if not self.__run:
#             raise RuntimeError('No phylofiltering model computed')
#         return self.__colnames
