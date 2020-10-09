# EELStoolbox
Toolbox for the automated analysis of large EELS datasets e.g. time series or spectrum images. The toolbox comprises several steps and lets you choose between different alignment methods, background models and deconvolution models. I hope it can help you with the automation of your data analysis. I have tested several datasets, but I greatly appreachiate feedback about problems with your specific dataset so that I can adjust the functions accordingly. 

Step 1:
Import the dataset. If you recorded a spectrum image using Gatan Digital Micrograph Version XYZ I recommend the xyz (matlab file) to convert the dataset to a python script.

Step 2:
Alignment of the dataset. In dependence of the nature of your dataset different methods are available. 
Method 1:
If your spectra contain the zero-loss peak (ZLP) you can directly use the ZLP to align you data. Run the following function:

Method 2:
If you recorded the spectra in Dual-Imaging mode you can use the ZLP spectra to align your high-loss data. Run the following function:

Method 3: 
If your data do not contain the ZLP, but all contain the same edge (whose shape doesn't change dramatically, e.g. C-K edge) you can use this edge to align your data using cross-correlation of your spectra. This requires a little more adjustements as described in the following. 
