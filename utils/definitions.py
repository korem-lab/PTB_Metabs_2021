import pyreadr
from os.path import join, exists
from pandas import read_pickle, read_excel

DATA_PATH = './data'
elovitz_ct_path = join(DATA_PATH, 'mm_no_Thermodesulfatator_v3.rda')
supp_tables_path = join(DATA_PATH, 'Supplementary Tables.xlsx')


## color schemes
ptb_colors = {'sPTB':'firebrick', 'TB':'dodgerblue'}
early_ptb_colors = {True:'violet', False:'navy'}
cst_colors = {'I':'k', 'II':'#F2E205', 'III':'#F28705', 'IV-A':'#003c8a', 'IV-B':'#05AFF2', 'V':'#A62103'}
mc_colors = {'A':'#000000', 'B':'#E69F00', 'C':'#F0E442', 'D':'#009E73', 'E':'#56B4E9', 'F':'#0072B2'}
race_colors = {1:'#E66100', 0:'#5D3A9B'}
