import os

from fig_1A import figure_1A
from fig_1B import figure_1B
from fig_1C import figure_1C
from fig_1DEFG import figures_1DEFG
from fig_2 import figure_2A, figure_2B, figure_2D
from fig_3 import figure_3A, figure_3B, figure_3C, figure_3D

from fig_S1 import figure_S1
from fig_S2 import figures_S2AB, figure_S2C, figures_S2DEF, figure_S2G
from fig_S3 import figures_S3ABEFG, figures_S3CD
from fig_S4 import figure_S4A, figure_S4B, figure_S4C, figure_S4D, figure_S4E
from fig_S5 import figure_S5B, figure_S5C, figure_S5D, figure_S5E, figure_S5F
from fig_S6 import figure_S6ABC, figure_S6D, figure_S6E
from fig_S8 import figures_S8ABD, figure_S8C
from fig_S9 import figures_S9AB, figure_S9C, figure_S9D


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

    figure_S1()
    figures_S2AB()
    figure_S2C()
    figures_S2DEF()
    figure_S2G()
    figures_S3ABEFG()
    figures_S3CD()
    figure_S4A()
    figure_S4B()
    figure_S4C()
    figure_S4D()
    figure_S4E()
    figure_S5B()
    figure_S5C()
    figure_S5D()
    figure_S5E()
    figure_S5F()
    figure_S6ABC()
    figure_S6D()
    figure_S6E()
    figures_S8ABD()
    figure_S8C()
    figures_S9AB()
    figure_S9C()
    figure_S9D()


if __name__ == "__main__":

    if not os.path.exists("./figurePanels"):
        os.makedirs("./figurePanels")

    if not os.path.exists("./microbe_metabolite_networks"):
        os.makedirs("./microbe_metabolite_networks")

    main()