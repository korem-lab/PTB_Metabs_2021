# PTB_Metabs_2021
Analyses pertaining to the manuscruipt Preterm birth is associated with xenobiotics 1 and predicted by the vaginal metabolome

William F. Kindschuh<sup>1*</sup>, Federico Baldini<sup>1*</sup>, Martin C. Liu<sup>1,2*</sup>, Jingqiu Liao<sup>1</sup>, Yoli Meydan<sup>1</sup>, 
Harry H.Lee<sup>1</sup>, Almut Heinken<sup>3</sup>, Ines Thiele<sup>3,4,5,6</sup>, Maayan Levy<sup>7</sup>&, Tal Korem<sup>1,8,9</sup>&

Author affiliations:

1 Program for Mathematical Genomics, Department of Systems Biology, Columbia University Irving Medical Center, New York, NY

2 Department of Biomedical Informatics, Columbia University Irving Medical Center, New York, NY

3 School of Medicine, National University of Ireland, Galway, Galway, Ireland

4 Discipline of Microbiology, National University of Galway, Galway, Ireland

5 Ryan Institute, National University of Galway, Galway, Ireland

6 APC Microbiome Ireland, University College Cork, Cork, Ireland

7 Department of Microbiology, University of Pennsylvania Perelman School of Medicine, Philadelphia, PA

8 Department of Obstetrics and Gynecology, Columbia University Irving Medical Center, New York, NY

9 CIFAR Azrieli Global Scholars program, CIFAR, Toronto, Canada

* - these authors contributed equally to this work

& - corresponding authors, Tal Korem (tk2829@cumc.columbia.edu) and Maayan Levy (maayanle@pennmedicine.upenn.edu)

# Instructions for running code:

To install required dependencies run the following command:
"conda env create -f ptb_metabs.yml"

To reproduce the analyses and panels run the following commands:
"python generate_figures.py"
"python generate_prediction_figures.py"

Note - in order to avoid having to regenerate the null distributions of p-values required for the msea (extended data figure 5d) you will need to unzip the zipped p-value files contained within data/msea 
