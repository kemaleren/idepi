import numpy as np
import Bio.PDB as biopdb
import Bio.SeqIO as seqio
from collections import defaultdict
from idepi.argument import parse_args
from idepi.argument import init_args, hmmer_args
from idepi.util import generate_alignment
from idepi.util import is_refseq
from idepi.util import set_util_params
from Bio import AlignIO
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.SeqUtils import seq3
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.base import BaseEstimator, TransformerMixin
from idepi.labeledmsa import LabeledMSA

import os

DATA_DIR = "/home/kemal/devel/idepi/data"
FASTA_FILE = os.path.join(DATA_DIR, '4NCO.fasta.txt')
PDB_FILE = os.path.join(DATA_DIR, '4NCO.pdb')


# read FASTA file
with open(FASTA_FILE, "rU") as handle:
    records = list(seqio.parse(handle, "fasta"))

FASTA_SEQ = records[0]

# read PDB file
parser = biopdb.PDBParser()
full_structure = parser.get_structure('gp140', PDB_FILE)
model = full_structure[0]

GP120S = list(model[i] for i in 'AEI')


def residue_center(r):
    return np.vstack(list(a.coord for a in r)).mean(axis=0)


def atom_distance(r1, r2):
    return min(a1 - a2 for a1 in r1 for a2 in r2)


def align_env(seq):
    parser, ns, args = init_args(description="align", args=[])
    parser = hmmer_args(parser)
    ARGS = parse_args(parser, [], namespace=ns)
    set_util_params(ARGS.REFSEQ.id)
    generate_alignment([records[0]], 'tmp.sto', is_refseq, ARGS, load=False)
    return AlignIO.read('./tmp.sto', 'stockholm')


def align(a, b):
    matrix = matlist.blosum62
    gap_open = -10
    gap_extend = -0.5
    alns = pairwise2.align.globalds(a, b, matrix, gap_open, gap_extend)
    return alns[0]


# assign coordinates of gp120 to fasta sequence
def make_seq_pdb_dicts(fasta_seq, pdb_structures):
    f2r_dict = defaultdict(list)
    r2f_dict = {}
    for struct in pdb_structures:
        ppb = biopdb.PPBuilder(radius=20)
        polys = ppb.build_peptides(struct)
        for poly in polys:
            poly_seq = poly.get_sequence()
            f_align, p_align, score, begin, end = align(fasta_seq, poly_seq)
            f_idx = -1
            p_idx = -1
            for i in range(begin, end):
                if f_align[i] != "-":
                    f_idx += 1
                if p_align[i] != "-":
                    p_idx += 1
                if f_align[i] != "-" and p_align[i] != "-":
                    residue = poly[p_idx]
                    f2r_dict[f_idx].append(residue)
                    r2f_dict[residue] = f_idx
    return f2r_dict, r2f_dict


def make_seqrecord_dicts(seq):
    seq_to_ref = {}
    ref_to_seq = {}
    seq_idx = -1
    for ref_idx in range(len(seq)):
        if seq[ref_idx] == "-":
            continue
        seq_idx += 1
        seq_to_ref[seq_idx] = ref_idx
        ref_to_seq[ref_idx] = seq_idx
    return seq_to_ref, ref_to_seq


def make_alignment_dicts(msa):
    assert len(msa) == 2
    seq_a, seq_b = msa
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


def find_nearby(i, f2r, r2f, searcher, radius):
    """for position i, find nearby positions"""
    if i not in f2r:
        return []
    nearby = set()
    for res in f2r[i]:
        found = searcher.search(residue_center(res), radius=radius,
                                level='R')
        nearby.update(r2f[r] for r in found if r in r2f)
    return nearby


class MSAVectorizerIsoelectric(BaseEstimator, TransformerMixin):

    def __init__(self, encoder, fasta_seq=None, gp120s=None,
                 radius=10):
        if not isinstance(radius, int) or radius < 0:
            raise ValueError('radius expects a positive integer')
        if fasta_seq is None:
            fasta_seq = FASTA_SEQ
        if gp120s is None:
            gp120s = GP120S
        self.__alignment_length = 0
        self.encoder = encoder
        self.fasta_seq = fasta_seq
        self.gp120s = gp120s
        self.radius = radius
        self.feature_names_ = []
        self.vocabulary_ = {}

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
        self.nearby = find_nearby(self.fasta_seq, self.f2r, self.r2f,
                                  self.searcher, radius=self.radius)

        # align env sequence and PDB fasta sequence
        self.aligned_env_pdb = align_env(self.fasta_seq)

        # make final dictionaries
        env_to_pdb, pdb_to_env = make_alignment_dicts(self.aligned_env_pdb)
        self.env_to_pdb = env_to_pdb
        self.pdb_to_env = env_to_pdb

    def feature_name(self, label):
        return 'isoelectric_label_{}_radius_{}'.format(label, self.radius)

    def fit(self, alignment):

        if not isinstance(alignment, LabeledMSA):
            raise ValueError("MSAVectorizers require a LabeledMSA")

        column_labels = list(alignment.labels)
        feature_names = []

        k = 0
        for i in range(alignment.get_alignment_length()):
            try:
                feature_name = self.feature_name(column_labels[i])
                feature_names.append(feature_name)
                k += 1
            except KeyError:
                pass

        self.__alignment_length = alignment.get_alignment_length()
        self.feature_names_ = feature_names

        return self

    def transform(self, alignment):
        ncol = alignment.get_alignment_length()

        if ncol != self.__alignment_length:
            msg = ('alignment length ({0:d}) does not match the'
                   'learned length ({1:d})'.format(
                       ncol,
                       self.__alignment_length))
            raise ValueError(msg)

        data = np.zeros((len(alignment), ncol), dtype=int)

        for i, seq in enumerate(alignment):
            # make forward and backward alignment dictionaries
            seq_to_ref, ref_to_seq = make_seqrecord_dicts(seq)
            seq_ = ''.join(ltr.upper() for ltr in str(seq.seq))
            for j in range(ncol):
                try:
                    ref_idx = j
                    pdb_idx = self.env_to_pdb[ref_idx]
                    nearby_pdb = find_nearby(pdb_idx, self.f2r, self.r2f,
                                             self.searcher, self.radius)
                    nearby_ref = set(self.pdb_to_env[elt]
                                     for elt in nearby_pdb)
                    nearby_seq = set(ref_to_seq[elt] for elt in nearby_ref)

                    nearby_seq.add(ref_to_seq[j])
                    residue_seq = ''.join(seq_[i] for i in nearby_seq)
                    analysis = ProteinAnalysis(residue_seq)
                    value = analysis.isoelectric_point()

                    data[i, j] = value
                except KeyError:
                    pass

        return data

    def get_feature_names(self):
        return self.feature_names_
