# Thesis_Data_Science_RTL_UvA
This is the repository containing all the notebooks with respect to data processing, model development, results, evaluation, and validation for my thesis research. 

This research is aimed at finding a way to replace humans in contextual tag generation for video content. 

The performance of several models are compared - ranging from State-of-the-Art to traditional machine learning.

Evaluation was done based on recall (how many generated tags are also in the ground truth sets on average), and on BERT-based cosine similarity scores. 

After all models have generated tag sets for a dataset, an experiment was done in which tagging experts had to choose the best tag set for each movie entry. This was done while not knowing which tag set belonged to what model. 

The full MPST dataset could not be uploaded to the large size. The dataset can be downloaded from: https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags/data. Same goes for the results files. 

A sample of the MPST data which the models used is included.

Samples of the results of each model are also included.
