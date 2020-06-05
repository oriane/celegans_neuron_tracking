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

J. P. Nguyen, A. N. Linder, G. S. Plummer, J. W. Shaevitz, and A. M. Leifer, “Automatically tracking neurons in a moving and deforming brain,” PLOS Computational Biology, vol. 13, no. 5, p. e1005517, May 2017, doi: 10.1371/journal.pcbi.1005517.

