 ## Emulation and calibration of Cardiac Digital Twins using machine learning

 ### Description
Digital twins (DTs) are mathematical or computer models tailored to specific instances of real world systems or processes. DTs are increasingly used in healthcare, engineering, manufacturing and industry to provide personalised recommendations, prognosis and future predictions. Due to the complexity of modelling real world systems at high resolution, DTs often comprise high dimensional or multi-scale mathematical modelling, making them computationally expensive and slow to run, meaning it can be hard to utilise them in fast-paced real world settings, such as clinical environments.

In this project we will focus on the development of tools to aid the emulation and calibration of cardiac DTs. Cardiac DTs have the capacity to be a key tool for personalised medicine, which will allow us to make data driven clinical recommendations in real time. However, sufficiently detailed cardiac models are often computationally expensive to run and this limits their use in fast-paced clinical settings.

This project will leverage machine learning techniques to emulate a cohort of cardiac patients. Then use these emulators as a surrogate to allow us to perform model calibration to learn unknown parameters using a Bayesian statistical approach. The cohort of calibrated models may then be used for patient forecasting and in-silico trials.

### Scientific Areas 
Complexities of human health and disease, including clinical and population-based approaches <br> 
Development of methodologies, conceptual frameworks, technologies, tools or techniques that could benefit health-related research

### Research Questions 
What are the most desirable methods of cardiac model emulation? <br> 
Is a single emulator the best choice for all patients? / Do anatomical differences affect the emulator choice? 

**Dataset 1** <br>
- Reaction-eikonal model of electrophysiology simulated on a virtual cohort of 19 adult healthy four-chamber heart meshes from CT images <br> 
- 6 inputs, 2 outputs, 180 simulations per mesh <br> 

  *Rodero, Cristobal et al. “Linking statistical shape models and simulated function in the healthy adult human heart.” PLoS computational biology vol. 17,4 e1008851. 15 Apr. 2021, doi:10.1371/journal.pcbi.1008851*

**Dataset 2** <br> 
- Passive atrial deformation simulated on 10 patient specific left atrial meshes generated using MRI imaging <br> 
- 9 inputs, 7 outputs , 200 simulations per mesh

### Methods and Models

**Gaussian Processes :** has a mean and a covariance function. The GP is treated as a prior distribution over possible functions and we learn a posterior approximation of the true function, allowing prediction with uncertainty.
 
**Gaussian Processes (Correlated) :** Similar to Gaussian Processes but assumes correlation between simulator outputs. 

**Radial Basis Function (RBF) :** RBFs measure the distance of an input point to a given centre, returning small values if the input is far from the centre and large values if the input is near the centre. In RBF Networks, the activation function for each node is an RBF.

**Multilayer Perceptron (MLP) :** Neural networks with an input layer, one or more hidden layer(s), an output layer and fully connected nodes with non-linear activation functions.

**Ensemble Multilayer Perceptron (E-MLP) :** Multiple independent MLPs with different starting points and data splits then average the results.

**Ensemble Multilayer Perceptron Dropout (E-MLP-D) :** MLP with dropout layers (hidden layers with a dropout rate). A subset of the nodes are deactivated (proportional to the dropout rate).

**Random Forest (RF) :** Uses multiple decision trees then averages the answers from each tree to make its final prediction.

### Results

**Best emulator overall:** Gaussian Processes outperform others on average. <br>
**Patient-specific performance:** Dataset 1 – GP best for all patients; Dataset 2 – best emulator varies by patient.

### Python Packages 
**AutoEmulate** (development and comparison of emulators) <br>
**GPyTorch** (Gaussian Processes) <br>
**UQPy** (uncertainty quantification)

**Funding :** Funded by the Wellcome Trust and completed at the University of Nottingham
