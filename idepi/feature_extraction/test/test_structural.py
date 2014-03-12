import unittest

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from idepi.feature_extraction import MSAVectorizerIsoelectric
from idepi.feature_extraction import MSAVectorizerDifference
from idepi.feature_extraction import MSAVectorizerGap
from idepi.feature_extraction import MSAVectorizerGapIsoelectric
import idepi.feature_extraction._msavectorizerstructural as struct


class TestIsoelectric(unittest.TestCase):
    def test_base(self):
        vectorizer = MSAVectorizerIsoelectric()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X != 0)
        # TODO: how to check correctness?


class TestDifference(unittest.TestCase):
    def test_base(self):
        vectorizer = MSAVectorizerDifference()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X == 0)

    def test_one_difference(self):
        seqstr = str(struct.FASTA_SEQ.seq)
        seqstr = seqstr[:100] + 'A' + seqstr[101:]
        seq = SeqRecord(Seq(seqstr))
        vectorizer = MSAVectorizerDifference()
        X = vectorizer.fit_transform((None, (seq,)))
        assert X.sum() > 1
        assert set(X.ravel()) == set((0, 1))


class TestGap(unittest.TestCase):
    def test_base(self):
        vectorizer = MSAVectorizerGap()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X == 0)

    def test_one_gap(self):
        seqstr = str(struct.FASTA_SEQ.seq)
        seqstr = seqstr[:100] + 'A' + seqstr[100:]
        seq = SeqRecord(Seq(seqstr))
        vectorizer = MSAVectorizerGap()
        X = vectorizer.fit_transform((None, (seq,)))
        assert X.sum() == 1
        assert X[0, 99] == 1


class TestGapIsoelectric(unittest.TestCase):
    def test_base(self):
        vectorizer = MSAVectorizerGapIsoelectric()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X == 0)

    def test_one_gap(self):
        seqstr = str(struct.FASTA_SEQ.seq)
        seqstr = seqstr[:100] + 'A' + seqstr[100:]
        seq = SeqRecord(Seq(seqstr))
        vectorizer = MSAVectorizerGapIsoelectric()
        X = vectorizer.fit_transform((None, (seq,)))
        analysis = ProteinAnalysis('A')
        assert X.sum() == analysis.isoelectric_point()
        assert X[0, 99] == analysis.isoelectric_point()


if __name__ == '__main__':
    unittest.main()
