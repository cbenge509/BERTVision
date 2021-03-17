import pandas as pd
import numpy as np
import csv

# load rte
rte = pd.read_csv('error_analysis_RTE.csv').sort_values(by='idx').reset_index(drop=True)
ap_rte = pd.read_csv('error_analysis_AP_RTE.csv').sort_values(by='idx').reset_index(drop=True)

# generate mask where values not equal
rte_mask = rte['pred'] != ap_rte['pred']

# view differences
ap_rte[rte_mask]

# view some of the original messages
rte_dev = pd.read_csv('C:\\w266\\data\\GLUE\\Recognizing Textual Entailment\\RTE\\dev.tsv',
                      sep='\t',
                      encoding='latin-1')
# specify dev cols
rte_dev.columns = ['id', 'sentence1', 'sentence2', 'label']
# relabel
rte_dev.label = np.where(rte_dev.label == 'entailment', 1, 0)

# randomly select a few for analysis
rte_errors = rte_dev[rte_mask].sample(frac=0.1)

rte_errors.to_csv('rte_sample_errors.csv', index=False)

# load msr
msr = pd.read_csv('error_analysis_MSR.csv').sort_values(by='idx').reset_index(drop=True)
ap_msr = pd.read_csv('error_analysis_AP_MSR.csv').sort_values(by='idx').reset_index(drop=True)

# generate mask where values not equal
msr_mask = msr['pred'] != ap_msr['pred']

# view differences
ap_msr[msr_mask]

# view some of the original messages
msr_dev = pd.read_csv('C:\\w266\\data\\GLUE\\Microsoft Research Paraphrase Corpus\\msr_paraphrase_test.txt', sep='\t',
                         encoding='latin-1',
                         error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)
# specify dev cols
msr_dev.columns = ['label', 'id', 'NoneField', 'sentence1', 'sentence2']

# randomly select a few for analysis
msr_errors = msr_dev[msr_mask].sample(frac=0.1)

msr_errors.to_csv('msr_sample_errors.csv', index=False)

#
