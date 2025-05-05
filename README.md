# Predicting-Multi-Class-Drug-Drug-Interactions


This is GitHub repo for the **Predicting-Multi-Class-Drug-Drug-Interactions** paper.

- In [Graph Embedding Experiments](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/Graph_Embedding_Experiments), are the experiment files for the Graph Embedding Experiments that utilize the Biomedical Literature Knowledge Graph using Neo4j.
- In [BLGPA Experiments](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/BGLPA_Experiments), you will find the Path Analysis (BLGPA) Experiment files along with the file derived from the BLGPA method. 


## How to replicate

### Graph Embedding Experiment


### BLGPA Experiment
1. Download from BLGPA folder, FinaFeatures.xlsx and BLGPA.ipynb
  - FinaFeatures.xlsx contains the dataset from the BLGPA pipeline as described in the paper.
  - BLGPA.ipynb is a notebook that performs the experiments    
2. Run the notebook.
3. The notebook creates the List-of-Predictions.csv, which contains the list of predicted (proposed) DDIs that have not yet been discovered.
  - The dictionary `param_grim` has 3 Random Forest parameters you can change to try other Grid Search combinations.
  - Sampling values contain pairs for [downsampling, upsampling].
    - The downsampling value divides the number of the majority class by that number.
    - The upsampling value multiples the number of minority classes by that number   

