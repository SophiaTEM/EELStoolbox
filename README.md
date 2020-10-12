# EELStoolbox
Toolbox for the automated analysis of large EELS datasets e.g. time series or spectrum images. The toolbox comprises several steps and lets you choose between different alignment methods, background models and deconvolution models. I hope it can help you with the automation of your data analysis. I have tested several datasets, but I greatly appreachiate feedback about problems with your specific dataset so that I can adjust the functions accordingly. 

### Step 1 - Import dataset:
If you recorded a spectrum image using Gatan Digital Micrograph Version XYZ I recommend the xyz (matlab file) to convert the dataset to a python script. The functions expect your data as an array with the form [xaxis, (yaxis,) spectrum]. 

### Step 2 - Alignment of the spectra:
Alignment of the dataset. In dependence of the nature of your dataset different methods are available. 
__Method 1:__
If your spectra contain the zero-loss peak (ZLP) you can directly use the ZLP to align you data. The function requires the dispersion (eV/channel) and a threshold value for the intensity of the ZLP maximum. 
```python
[EELSaligned, energyscale, appliedShifts] = MWf.ZLPalignment(EELSdata, Dispersion, IntensityZLP)
print s
```

__Method 2:__
If you recorded the spectra in Dual-Imaging mode you can use the ZLP spectra to align your high-loss data.
```python
[EELSaligned, energyscale, appliedShifts] = MWf.ZLPalignment(EELSdata, Dispersion, IntensityZLP)
print s
```

__Method 3:__ 
If your data do not contain the ZLP, but all contain the same edge (whose shape doesn't change dramatically, e.g. C-K edge) you can use this edge to align your data using cross-correlation of your spectra. This requires a little more adjustements as described in the following. 

### Step 3 - Background correction:
Different models to subtract the background from the EELS dataset are available. The powerlaw model works best for most high-loss edges, while the polynomial models fit low-loss data the best. The mathematic formulas are summarized in the following equation together with the respective literature reference.  

