# Predicting-Multi-Class-Drug-Drug-Interactions

This is GitHub repo for the **Predicting-Multi-Class-Drug-Drug-Interactions** paper.

- In [Graph Embedding Experiments](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/Graph_Embedding_Experiments), are the experiment files for the Graph Embedding Experiments that utilize the Biomedical Literature Knowledge Graph using Neo4j.
- In [BLGPA Experiments](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/BGLPA_Experiments), you will find the Path Analysis (BLGPA) Experiment files along with the file derived from the BLGPA method. 
- In [RGCN Experiments](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/RGCN-Experiments), are the experiment files for the RGCN model that utilize csv files that must be extracted from Neo4j. 


## How to replicate

### Graph Embedding Experiment
1. Start a connection to your Neo4j database containing the Biomedical Graph Database.
2. Open one of the Embedding-Experiments python files.
3. Choose the embedding you want to train, in the `embed` variable.
4. The `data` variable reads a csv file containg the Drug (CUI) Pairs and their interaction.
5. The code then trains the embeddings, saves them, loads them. Then it performs the nested cross validation and trains the classifier. 

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

### RGCN Experiment
1. Start Neo4j and extract all required csv files:
   - entities.tsv (including neo4j id, cui and sem_types)
   - articles.tsv (including neo4j id, pubmed/pmc ids)
   - relations_articles.tsv (including article-entity ids, relation_type and number of references)
   - relations_entities.tsv (including entity-entity ids, relation_type and number of references)
   - relations.tsv (union of the two files above)
2. Move the csv files in the same folder with python file.
3. Transform (CUIs to to Neo4j ids) the groundtruth file (ddi-taxonomy.csv) into two files for positive classes and negative class respectively:
   - posGroundtruth_filtered.tsv
   - negGroundtruth_filtered.tsv
5. Run the notebook.
 

## Knowledge Graph Generation
To regenerate the open data knowledge graph for lung cancer, use the software modules provided in  [Knowledge Graph Generation](https://github.com/MariosVottas/Predicting-Multi-Class-Drug-Drug-Interactions/tree/main/Knowledge%20Graph) and follow the instructions provided in this work:
https://arxiv.org/abs/1912.08633
