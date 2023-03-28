import os

from fig_1A import figure_1A
from fig_1B import figure_1B
from fig_1C import figure_1C
from fig_1DEFG import figures_1DEFG
from fig_2 import figure_2A, figure_2B, figure_2D
from fig_3 import figure_3A, figure_3B, figure_3C, figure_3D

from ext_data_fig_1 import ext_data_figure_1
from ext_data_fig_2 import ext_data_fig_2AB, ext_data_fig_2C, ext_data_fig_2D
from ext_data_fig_3 import ext_data_fig_3AB, ext_data_fig_3C, ext_data_fig_3DEF, ext_data_fig_3G
from ext_data_fig_4 import ext_data_fig_4ABEFG, ext_data_fig_4CD
from ext_data_fig_5 import ext_data_fig_5A, ext_data_fig_5B, ext_data_fig_5C, ext_data_fig_5D, ext_data_fig_5E, ext_data_fig_5F
from ext_data_fig_6 import ext_data_fig_6B, ext_data_fig_6C, ext_data_fig_6D, ext_data_fig_6E, ext_data_fig_6F
from ext_data_fig_7 import ext_data_fig_7ABC, ext_data_fig_7D, ext_data_fig_7E


def main():

    figure_1A()
    figure_1B()
    figure_1C()
    figures_1DEFG()
    figure_2A()
    figure_2B()
    figure_2D()
    figure_3A()
    figure_3B()
    figure_3C()
    figure_3D()

    ext_data_figure_1()
    ext_data_fig_2AB()
    ext_data_fig_2C()
    ext_data_fig_2D()
    ext_data_fig_3AB()
    ext_data_fig_3C()
    ext_data_fig_3DEF()
    ext_data_fig_3G()
    ext_data_fig_4ABEFG()
    ext_data_fig_4CD()
    ext_data_fig_5A()
    ext_data_fig_5B()

    if os.path.exists("data/vp.csv"):
        ext_data_fig_5C()
    else:
        print("data/vp.csv not found, skipping extended data figure 5C")

    ext_data_fig_5D()
    ext_data_fig_5E()
    ext_data_fig_5F()
    ext_data_fig_6B()
    ext_data_fig_6C()
    ext_data_fig_6D()
    ext_data_fig_6E()
    ext_data_fig_6F()
    ext_data_fig_7ABC()
    ext_data_fig_7D()
    ext_data_fig_7E()


if __name__ == "__main__":

    if not os.path.exists("./figurePanels"):
        os.makedirs("./figurePanels")

    if not os.path.exists("./microbe_metabolite_networks"):
        os.makedirs("./microbe_metabolite_networks")

    if not os.path.exists("./differential_abundance_sig_hits"):
        os.makedirs("./differential_abundance_sig_hits")

    main()