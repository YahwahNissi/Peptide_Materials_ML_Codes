import sys
from rdkit import Chem

def get_logP(peptide):
    logP = None
    norm_min = len(peptide.split('-'))*-2.09
    norm_max = len(peptide.split('-'))*3.64

    for pep in peptide.split('-'):
        if logP is None:
            logP = peptide_logp[pep]
        else:
            logP += peptide_logp[pep]

    norm_logP = (logP - norm_min)/(norm_max - norm_min)
    return logP, norm_logP


def get_abs_logP(peptide):
    logP = None
    norm_min = len(peptide.split('-'))*3.64
    norm_max = len(peptide.split('-'))*0

    for pep in peptide.split('-'):
        if logP is None:
            logP = peptide_logp[pep]
        else:
            logP += peptide_logp[pep]

    norm_logP = (abs(logP) - norm_min)/(norm_max - norm_min)
    return abs(logP), norm_logP


def get_norm_AP(AP, norm_min=0.97, norm_max=2.7):
    norm_AP = (AP - norm_min)/(norm_max - norm_min)
    return norm_AP


def get_norm_bscore(pep, bmin=0.4, bmax=2.0):
    bscore = 0
    for p in pep:
        bscore += peptide_betascore[p]
    norm_bscore = (bscore - len(pep)*bmin)/(len(pep)*bmax - len(pep)*bmin)

    return norm_bscore


def get_short_sequence(peptide):
    short_pep = ''
    for p in peptide:
        short_pep = short_pep + peptide_shortSeq[p]
    return short_pep


def get_long_sequence(pep):
    peptide = []
    for p in pep:
        peptide.append(peptide_longSeq[p])
    return '-'.join(peptide)


def get_smiles(pep):
    m = Chem.rdmolfiles.MolFromSequence(pep)
    return Chem.MolToSmiles(m)    


def get_single_smile_pg_fp(smile):

    try:
        m = Chem.MolFromSmiles(smile)

        if m is not None:
            afp = fp.v2_atom(smiles_string=smile, debug=1, polymer_fp_type='aT', ismolecule=1)
            mfp = fp.v2_moleculardescriptor(smiles_string=smile, debug=1, ismolecule=1)
            efp = fp.v2_extendeddescriptor(smile, debug=1, ismolecule=1)

            smile_fp = {'smiles': smile, **afp,**mfp,**efp}
            #print('Success, Fingerprinting ', smile)

    except:
        print('Failed Fingerprinting for ', smile)
        smile_fp = None

    return smile_fp


def get_peptide_charge(peptide):
    abs_charge = 0
    net_charge = 0
    pattern = ''
    
    for p in peptide.split('-'):
        abs_charge += abs(peptide_charge[p])
        net_charge += peptide_charge[p]

        pattern +='n' if peptide_charge[p]==0 else 'p'
        
    return net_charge, abs_charge, pattern



peptide_logp = {'ILE': -1.12,
                'LEU': -1.25,
                'PHE': -1.71,
                'VAL': -0.46,
                'MET': -0.67,
                'PRO': 0.14,
                'TRP': -2.09,
                'THR': 0.25,
                'GLN': 0.77,
                'CYS': -0.02,
                'TYR': -0.71,
                'ALA': 0.5,
                'SER': 0.46,
                'ASN': 0.85,
                'ARG': 1.81,
                'GLY': 1.15,
                'LYS': 2.8,
                'GLU': 3.63,
                'HIS+': 2.33,
                'ASP': 3.64,
                'GLU0': 0.11,
                'HIS': 0.11}

peptide_betascore = {'V': 2.0,
                     'I': 1.79,
                     'F': 1.4,
                     'Y': 1.37,
                     'C': 1.36,
                     'W': 1.23,
                     'T': 1.21,
                     'L': 1.15,
                     'M': 1.01,
                     'H': 0.99,
                     'R': 0.85,
                     'S': 0.81,
                     'K': 0.76,
                     'A': 0.75,
                     'Q': 0.72,
                     'G': 0.67,
                     'E': 0.65,
                     'N': 0.63,
                     'D': 0.55,
                     'P': 0.4}

peptide_shortSeq = {'ARG': 'R',
                'HIS': 'H',
                'LYS': 'K',
                'ASP': 'D',
                'GLU': 'E',
                'SER': 'S',
                'THR': 'T',
                'ASN': 'N',
                'GLN': 'Q',
                'CYS': 'C',
                'GLY': 'G',
                'PRO': 'P',
                'ALA': 'A',
                'ILE': 'I',
                'LEU': 'L',
                'MET': 'M',
                'PHE': 'F',
                'TRP': 'W',
                'TYR': 'Y',
                'VAL': 'V'}

peptide_longSeq = {'R': 'ARG',
                 'H': 'HIS',
                 'K': 'LYS',
                 'D': 'ASP',
                 'E': 'GLU',
                 'S': 'SER',
                 'T': 'THR',
                 'N': 'ASN',
                 'Q': 'GLN',
                 'C': 'CYS',
                 'G': 'GLY',
                 'P': 'PRO',
                 'A': 'ALA',
                 'I': 'ILE',
                 'L': 'LEU',
                 'M': 'MET',
                 'F': 'PHE',
                 'W': 'TRP',
                 'Y': 'TYR',
                 'V': 'VAL'}

peptide_charge =  {'ILE': 0,
        'LEU': 0,
        'PHE': 0,
        'VAL': 0,
        'MET': 0,
        'PRO': 0,
        'TRP': 0,
        'THR': 0,
        'GLN': 0,
        'CYS': 0,
        'TYR': 0,
        'ALA': 0,
        'SER': 0,
        'ASN': 0,
        'GLY': 0,
        'ARG': 1,
        'HIS': 1,        
        'LYS': 1,
        'GLU': -1,
        'ASP': -1}                     
