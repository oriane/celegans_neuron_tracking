# Neuron_Tracking

## To Install

Make sure that modules in requirement.txt are installed.

The module gmmreg has to be installed mannualy, instruction can be found at https://github.com/bing-jian/gmmreg-python (Installation instructions in the READ ME of the Repository.)

## To Run

Create two folder at root level: data and produced. 

Csv file of the segmentation to be tracked has to be in the data folder.

***Tracking*** notebook shows example of run of the algorithm. 

To be able to use the nd2 plot, nd2 file have to be exported to tif in a folder name data/{name_of_file}\_exported.
## Reference
Tracking Process has been adapted from a paper by Leifer and al. :

@article{nguyen2017automatically,
  title={Automatically tracking neurons in a moving and deforming brain},
  author={Nguyen, Jeffrey P and Linder, Ashley N and Plummer, George S and Shaevitz, Joshua W and Leifer, Andrew M},
  journal={PLoS computational biology},
  volume={13},
  number={5},
  year={2017},
  publisher={Public Library of Science}
}

Gmm algorithm is taken from: 

@article{Jian&Vemuri_pami11,
  author  = {Bing Jian and Baba C. Vemuri},
  title   = {Robust Point Set Registration Using {Gaussian} Mixture Models},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
  year = {2011},
  volume = {33},
  number = {8},
  pages = {1633-1645},
  url = {https://github.com/bing-jian/gmmreg/},
}
