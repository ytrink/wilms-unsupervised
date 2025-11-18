# wilms-unsupervised
code to reproduce figures from "Characterization of Continuous Transcriptional Heterogeneity in High-Risk Blastemal-Type Wilms' Tumors Using Unsupervised Machine Learning "

This paper presents an approach to the problem of modeling tumor heterogeneity by using the tandem of archetype analysis and topic modeling. 
First, we use the ParTI method (https://github.com/AlonLabWIS/ParTI) to compute gene expression archetypes. 
We then fit a generative topic model using a non-negative matrix factorization algorithm (https://arxiv.org/abs/2105.13440). The number of archetypes inferred provides a good clue for the number of topics to choose in the model fit.

This paper demonstrates this approach on a publicly available dataset from high-risk Wilms' tumors.

For questions and access to gene expression datasets, please email trinkya@biu.ac.il.
