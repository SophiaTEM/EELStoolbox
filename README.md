# EELStoolbox
Toolbox for the automated analysis of large EELS datasets e.g. time series or spectrum images. The toolbox comprises several steps and lets you choose between different alignment methods, background models and deconvolution models. I hope it can help you with the automation of your data analysis. 

Step 1:
Import the dataset. If you recorded a spectrum image using Gatan Digital Micrograph Version XYZ I recommend the xyz (matlab file) to convert the dataset to a python script.

Step 2:
Alignment of the dataset. In dependence of the nature of your dataset different methods are available. 
If your spectra contain the zero-loss peak (ZLP) you can directly use the ZLP to align you data (method 1).
