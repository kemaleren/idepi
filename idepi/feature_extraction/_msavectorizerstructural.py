"""Computes isoelectric point in radius around residues.

TODO:
-----
- variables names are inconsistent and confusing.

- generalize this code so that it can handle more than just PDB
  structure 4NCO.

- how to deal with gaps? right now we just ignore them.

"""

import os
from collections import defaultdict

import numpy as np

import Bio.PDB as biopdb
import Bio.SeqIO as seqio
from Bio.SeqUtils import seq3
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.base import BaseEstimator, TransformerMixin

from BioExt.scorematrices import HIV_BETWEEN_F
from BioExt.align import Aligner
from BioExt.misc import translate

from idepi import __path__ as idepi_path

# TODO: get rid of all these global variables.

DATA_DIR = os.path.join(idepi_path[0], 'data')
FASTA_FILE = os.path.join(DATA_DIR, '4NCO.fasta.txt')
PDB_FILE = os.path.join(DATA_DIR, '4NCO.pdb')


# read FASTA file and get gp120 sequence
with open(FASTA_FILE, "rU") as handle:
    records = list(seqio.parse(handle, "fasta"))
FASTA_SEQ = records[0]

# read PDB file and get gp120 chains
PARSER = biopdb.PDBParser()
FULL_STRUCTURE = PARSER.get_structure('gp140', PDB_FILE)
MODEL = FULL_STRUCTURE[0]
GP120S = list(MODEL[i] for i in 'AEI')


# TODO: the following dict-making functions are all similar. Find some
# way of abstracting out their core functionality.

def make_seq_pdb_dicts(fasta_seq, pdb_structures):
    """Align the FASTA sequence of gp120 to the PDB residues.

    Each position in the FASTA sequence corresponds to three residues,
    one from each strand of the trimer.

    Parameters
    ----------
    fasta_seq : SeqRecord
        Sequence of the gp120 protein.

    pdb_structures : iterable of Bio.PDB chains
        The three gp120 chains of the trimer.

    Returns
    ------
    f2r : dict
        f2r[i] is the set of residues corresponding to the `i`th residue in
        the fasta sequence.

    r2f : dict
        r2f[r] is the index of the fasta sequence corresponding to residue r.

    """
    f2r_dict = defaultdict(list)
    r2f_dict = {}
    aligner = Aligner(HIV_BETWEEN_F.load(), do_codon=False)

    for struct in pdb_structures:
        ppb = biopdb.PPBuilder(radius=20)
        polys = ppb.build_peptides(struct)
        for poly in polys:
            poly_seq = poly.get_sequence()
            score, f_align, p_align = aligner(fasta_seq, poly_seq)
            assert len(f_align) == len(p_align)
            f_idx = -1
            p_idx = -1
            for i in range(len(f_align)):
                if f_align[i] != "-":
                    f_idx += 1
                if p_align[i] != "-":
                    p_idx += 1
                if f_align[i] != "-" and p_align[i] != "-":
                    residue = poly[p_idx]
                    f2r_dict[f_idx].append(residue)
                    r2f_dict[residue] = f_idx
    return f2r_dict, r2f_dict


def make_alignment_dicts(seq_a, seq_b):
    """Map corresponding positions between the two sequences.

    Parameters
    ----------
    seq_a, seq_b : SeqRecord
        Aligned sequences.

    Returns
    -------
    a2b : dict
        Position ``i`` in the original ``seq_a`` maps to position
        ``a2b[i]`` in the original ``seq_b``.

    b2a : dict
        Position ``i`` in the original ``seq_b`` maps to position
        ``b2a[i]`` in the original ``seq_a``.

    """
    d_a_to_b = {}
    d_b_to_a = {}
    idx_a = -1
    idx_b = -1
    for i in range(len(seq_a)):
        if seq_a[i] != "-":
            idx_a += 1
        if seq_b[i] != "-":
            idx_b += 1
        if seq_a[i] != "-" and seq_b[i] != "-":
            d_a_to_b[idx_a] = idx_b
            d_b_to_a[idx_b] = idx_a
    return d_a_to_b, d_b_to_a


def residue_center(r):
    """the mean of the residue's atom coordinates"""
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def find_nearby(i, f2r, r2f, searcher, radius):
    """For residue in position i, find positions nearby residues.

    Parameters
    ----------
    i : int
        Position of a residue in the FASTA sequence.

    f2r, r2f : dicts
        Results of make_seq_pdb_dicts().

    searcher : Bio.PDB.NeighborSearch
        A searcher initialized with all the residues of interest.

    radius : int
        Distance in Angstroms.

    """
    nearby = set()
    if i not in f2r:
        return nearby
    for res in f2r[i]:
        found = searcher.search(residue_center(res), radius=radius,
                                level='R')
        nearby.update(r2f[r] for r in found if r in r2f)
    return nearby


class MSAVectorizerStructural(BaseEstimator, TransformerMixin):
    """Structural features.

    Uses the 3D structure of gp120 (derived from PDB structure id
    4NCO) to find nearby residues within some radius.

    For each transformed sequence, the final vector has dimensionality
    equal to the length of the reference hxb2 Env sequence.

    Residues are mapped to 3D coordinates through the following chain
    of alignments:

    target sequence <=> hxb2 Env <=> 4NCO FASTA file <=> 4NCO PDB file

    Parameters
    ----------
    fasta_seq : SeqRecord
        The sequence of the gp120 proteins.

    gp120s : iterable of Bio.PDB chains
        The three gp120 protein structures.

    radius : int
        Radius in Angstroms.

    """

    def __init__(self, fasta_seq=None, gp120s=None, radius=10):
        if not isinstance(radius, int) or radius < 0:
            raise ValueError('radius expects a positive integer')
        if fasta_seq is None:
            fasta_seq = FASTA_SEQ
        if gp120s is None:
            gp120s = GP120S
        self.__alignment_length = 0

        self.fasta_seq = fasta_seq
        self.gp120s = gp120s
        self.radius = radius
        self.feature_names_ = []

        self.atoms = list(a for chain in gp120s for a in chain.get_atoms())
        self.residues = list(r for chain in gp120s for r in chain)

        f2r, r2f = make_seq_pdb_dicts(fasta_seq, gp120s)

        for k, v in r2f.items():
            # TODO: this should not necessarily be true. There might be point
            # mismatches.
            assert k.resname.upper() == seq3(fasta_seq[v]).upper()

        for k, v in f2r.items():
            assert len(v) == 3

        self.f2r = f2r
        self.r2f = r2f

        self.searcher = biopdb.NeighborSearch(self.atoms)

    def feature_name(self, label):
        return '{}_label_{}_radius_{}'.format(self.name, label, self.radius)

    def fit(self, tofit):
        alignment, seqrecords = tofit
        self.ref_len = len(self.fasta_seq)
        column_labels = list(map(str, range(self.ref_len)))
        feature_names = []
        k = 0

        # TODO: these indices refer to 4NCO gp120.
        for i in range(self.ref_len):
            try:
                feature_name = self.feature_name(column_labels[i])
                feature_names.append(feature_name)
                k += 1
            except KeyError:
                pass

        self.__alignment_length = len(self.fasta_seq)
        self.feature_names_ = feature_names

        return self

    def _compute(self, seq, seq_to_ref, ref_to_seq, ref_idx,
                 nearby_ref, nearby_seq):
        # TODO: cut down on the number of arguments
        raise Exception('this is a base class only')

    def transform(self, tofit):
        alignment, seqrecords = tofit
        ncol = self.ref_len

        if ncol != self.__alignment_length:
            msg = ('alignment length ({0:d}) does not match the'
                   'learned length ({1:d})'.format(
                       ncol,
                       self.__alignment_length))
            raise ValueError(msg)

        # TODO: this assumes all default values are zeros
        data = np.zeros((len(seqrecords), ncol), dtype=int)
        aligner = Aligner(HIV_BETWEEN_F.load(), do_codon=False)

        # TODO: check before doing this
        seqrecords = list(translate(s) for s in seqrecords)

        for i, seq in enumerate(seqrecords):
            _, seq_a, seq_b = aligner(seq, self.fasta_seq)
            seq_to_ref, ref_to_seq = make_alignment_dicts(seq_a, seq_b)
            for j in range(ncol):
                try:
                    # TODO: the first half of this can be precomputed
                    # during fit() and not all of it needs to be
                    # computed here
                    ref_idx = j
                    nearby_ref = find_nearby(ref_idx, self.f2r, self.r2f,
                                             self.searcher, self.radius)
                    nearby_ref.add(j)
                    nearby_seq = set(ref_to_seq[elt] for elt in nearby_ref)
                    data[i, j] = self._compute(seq, seq_to_ref,
                                               ref_to_seq, ref_idx,
                                               nearby_ref, nearby_seq)
                except KeyError:
                    pass
        return data

    def get_feature_names(self):
        return self.feature_names_


class MSAVectorizerIsoelectric(MSAVectorizerStructural):
    """Computes the isoelectric point of nearby residues."""
    name = 'isoelectric'

    def feature_name(self, label):
        return 'isoelectric_label_{}_radius_{}'.format(label, self.radius)

    def _compute(self, seq, seq_to_ref, ref_to_seq, ref_idx,
                 nearby_ref, nearby_seq):
        residue_seq = ''.join(seq[i].upper() for i in nearby_seq)
        analysis = ProteinAnalysis(residue_seq)
        return analysis.isoelectric_point()


class MSAVectorizerDifference(MSAVectorizerStructural):
    """Computes the number of residues that differ from the reference."""
    name = 'difference'

    def _compute(self, seq, seq_to_ref, ref_to_seq, ref_idx,
                 nearby_ref, nearby_seq):
        if not nearby_seq:
            return -1  # FIXME: probably not an appropriate value for this case
        result = 0
        for seq_idx in nearby_seq:
            seq_residue = seq[seq_idx]
            ref_residue = self.fasta_seq[seq_to_ref[seq_idx]]
            if seq_residue.upper() == ref_residue.upper():
                result += 1
        return result
