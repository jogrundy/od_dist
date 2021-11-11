# Outlier Data Detection Systems - ODDS

# As used in paper "Simple Models are Effective in Anomaly Detection in Multi-variate Time Series"

> pip install odds

THe work is done by the **OD** object. Import the 'OD' object as follows:

> from odds import OD

Instantiate the object with the 'algo' argument, where a short string represents the algorithm you wish to use. In this case, 'VAR' refers to vector autoregression, a simple linear multidimensional regression algorithm. Other implemented algorithms are listed below.

> od = OD('VAR')

To use the object, you need to call the 'get_os()' function, with 'X' as its argument, where X is a data matrix, **n** samples by **p** features. **p must be 2 or greater to work**
on many of the systems, this returns a vector with n scores, one for each sample.

> outlier_scores = od.get_os(X)

The higher scores are the more outlying. you can then set a threshold if you wish, or just look at the ranking. Scores have not been sanitised, they may contain 'nan' values particularly from the 'VAE' if the data input has not been scaled. However it seems other algorithms work better without scaling, so inputs are not scaled. 


Valid strings for outlier algorithms:

- 'VAR' vector autoregression
- 'FRO' ordinary feature regression
- 'FRL' LASSO feature regression
- 'FRR' Ridge feature regression
- 'GMM' Gaussian Mixture model
- 'IF' isolation Forest
- 'DBSCAN' Density Based Spatial clustering and noise
- 'OCSVM' one class support vector machine
- 'LSTM' long short term memory
- 'GRU' gated recurrent unit
- 'AE' autoencoder
- 'VAE' variational autoencoder
- 'OP' outlier pursuit
- 'GOP' graph regularised outlier pursuit
- 'RAND' random scoring (for baseline comparison)
