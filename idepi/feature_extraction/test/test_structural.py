import unittest

import numpy as np

from idepi.feature_extraction import MSAVectorizerIsoelectric
from idepi.feature_extraction import MSAVectorizerDifference
import idepi.feature_extraction._msavectorizerstructural as struct


class TestBaseCase(unittest.TestCase):
    def test_isoelectric(self):
        vectorizer = MSAVectorizerIsoelectric()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X != 0)
        # TODO: how to check correctness?

    def test_difference(self):
        vectorizer = MSAVectorizerDifference()
        X = vectorizer.fit_transform((None, (struct.FASTA_SEQ,)))
        assert np.all(X == 0)


if __name__ == '__main__':
    unittest.main()
