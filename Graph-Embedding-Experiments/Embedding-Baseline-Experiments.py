# Define Neo4j connections
from neo4j import GraphDatabase
import pandas as pd
from sklearn import metrics
from pykeen.triples import TriplesFactory
import torch
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List
import pykeen.nn
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter


# Connect to Neo4j 
print('Neo4j connection details...')
host = 'bolt://localhost:7687'
user = 'neo4j'
password = 'iasis'
driver = GraphDatabase.driver(host,auth=(user, password))



# Insert Neo4j data into a Dataframe
def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

# Get all IDs
def getIdFromCUI(cui): 
    cquery = r"MATCH (s:Entity) WHERE (s.id='"+cui+ r"') RETURN toString(id(s)) as id"
    loc_id = run_query(cquery)
    if len(loc_id)==0:
        #print('no id for cui: '+cui)
        return "null"
    return loc_id['id'][0]

# Embedding method
embed = 'TransE' # change with 'HoLE', 'DistMult', 'RESCAL'

########### FIRST CREATE DICTIONARY + MODEL ##################
print('FIRST CREATE DICTIONARY + MODEL')

# Cypher query to get ALL the database
data = run_query("""
MATCH (s)-[r]->(t)
RETURN toString(id(s)) as source, toString(id(t)) AS target, type(r) as type
""")

from pykeen.triples import TriplesFactory

print('Insert neo4j graph into pykeen and save dictionary...')
###Insert neo4j graph into pykeen
tf = TriplesFactory.from_labeled_triples(
  data[["source", "type", "target"]].values,
  create_inverse_triples=False,
  entity_to_id=None,
  relation_to_id=None,
  compact_id=False,
  filter_out_candidate_inverse_relations=True,
  metadata=None,
)
#tf.to_path_binary(embed+'_dictionary.pt')
print('Done! ')


print('Now, dividing pykeen data into training-test-validation.')

# This saves the model

training, testing, validation = tf.split([.8, .1, .1])

print('Now preparing embeddings...')

from pykeen.pipeline import pipeline

print('training ', embed)
result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=embed,
        stopper='early',
        epochs=100,
        dimensions=100,
        random_seed=42,
        device='cuda:2'
)
print(embed, ': Saving created model to a local file...')
result.save_to_directory(embed)
print('Done')


###########THEN LOAD AND RUN k-FOLD CV
from pykeen.triples import TriplesFactory

print('loading from the file to get link predictions.') 
model = torch.load(embed+'/trained_model.pkl')
#tf=TriplesFactory.from_path_binary(embed+'_dictionary.pt') # this loads dict

print('Done')


relID=tf.relation_to_id["INTERACTS_WITH"]
relation: torch.IntTensor = torch.as_tensor(relID, device='cuda:2')
relation_representation_modules: List['pykeen.nn.Representation']  = model.relation_representations
relation_embeddings: pykeen.nn.Embedding  = relation_representation_modules[0]
relation_embedding_tensor: torch.FloatTensor  = relation_embeddings(indices=relation)
rel_embedding = Variable(relation_embedding_tensor).cpu()
print('relation interacts_with embedding_tensor: ')
#print(rel_embedding.data[:])
#print(entity_embedding_tensor)

#check3 = input("pause...")
# Insert Ground Truth
data = pd.read_csv("/home/faisopos/workspace/marios/CUIs.csv")
cuiPairs=data["CUI_PAIR"]
pairs_ground=data["INTERACTS"]

print('create RF Classifier...')
#Create a RF Classifier
classifier=RandomForestClassifier() 
 
k=10
precision = np.zeros(k)
recall = np.zeros(k)
f1=np.zeros(k)
print('Prepare the cross-validation procedure.')
cv = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
f=0

number_of_classes = len(np.unique(pairs_ground))

print("This is the number of classes", number_of_classes)
# Lists to store class-wise metrics
precision_per_class = np.zeros((k, number_of_classes))
recall_per_class = np.zeros((k, number_of_classes))
f1_per_class = np.zeros((k, number_of_classes))


print('Start '+str(k)+'-fold CV...')
fold_number = 0
for train_index, test_index in cv.split(cuiPairs,pairs_ground):
    fold_number += 1
    print(f"fold_number:{fold_number}\n")

    c_trainPairs, c_testPairs= cuiPairs.iloc[train_index], cuiPairs.iloc[test_index] 
    y_train, y_test = pairs_ground.iloc[train_index], pairs_ground.iloc[test_index]


    X_train = c_trainPairs.values.reshape(-1,1)
    c_train0 = pd.DataFrame(X_train)
    c_train = c_train0[0]  

    print("Now get node embeddings for each pair")
    entity_representation_modules: List['pykeen.nn.Representation']  = model.entity_representations
    entity_embeddings: pykeen.nn.Embedding  = entity_representation_modules[0]

    rows, cols = (len(y_train), 3*len(rel_embedding))
    X_train = [[0 for i in range(cols)] for j in range(rows)]
    for i,trainPair in enumerate(c_train):
        cuiList = trainPair.split("_");
        drCUI=cuiList[0]
        tarCUI= cuiList[1]
        tarId = getIdFromCUI (tarCUI)
        #print("this is the tarID", tarId)
        if tarId is None or tarId == "null":
            continue
        drId = getIdFromCUI (drCUI)
        #print("this is the drId", drId)
        if drId is None or drId== "null":
            continue
        dr_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[drId], device='cuda:2')
        dr_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=dr_entity)
        dr_embedding = Variable(dr_entity_embedding_tensor).cpu()

        tar_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[tarId], device='cuda:2')
        tar_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=tar_entity)
        tar_embedding = Variable(tar_entity_embedding_tensor).cpu()
        X_train[i]=np.concatenate((dr_embedding, rel_embedding, tar_embedding), axis=None)
        #print(f"This is the X_train_i:{X_train[i]}")

    print("classifier fit...")
    
    # Train the classifier
    classifier.fit(X_train, y_train)

    print("Now get node embeddings for each TEST pair")
    rows, cols = (len(y_test), 3*len(rel_embedding))
    X_test = [[0 for i in range(cols)] for j in range(rows)]
    
    # Make predictions
    y_pred = []
    y_test_fold = []
    i_counter = 0
    for i,testPair in enumerate(c_testPairs):
        cuiList = testPair.split("_");
        drCUI=cuiList[0]
        tarCUI= cuiList[1]
        drId = getIdFromCUI (drCUI)
        if drId is None or drId == "null":
            continue   
        dr_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[drId], device='cuda:2')
        dr_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=dr_entity)
        dr_embedding = Variable(dr_entity_embedding_tensor).cpu()

        tar_entity: torch.IntTensor = torch.as_tensor(tf.entity_to_id[tarId], device='cuda:2')
        tar_entity_embedding_tensor: torch.FloatTensor  = entity_embeddings(indices=tar_entity)
        tar_embedding = Variable(tar_entity_embedding_tensor).cpu()

        X_test[i]= np.concatenate((dr_embedding, rel_embedding, tar_embedding), axis=None)
        X_test[i] = X_test[i].reshape(1, -1) 

        y_pred.append(classifier.predict(X_test[i]))
        y_test_fold.append(y_test.iloc[i]) 

    for class_label in range(number_of_classes):

        precision_per_class[f, class_label] = precision_score(y_test_fold, y_pred, average='macro')
        recall_per_class[f, class_label] = recall_score(y_test_fold, y_pred, average='macro')
        f1_per_class[f, class_label] = f1_score(y_test_fold, y_pred, average='macro')

        print(f"This is the precision:{precision_per_class[f, class_label]} of fold:{f} and class{class_label}\n")

    f += 1 


# Print class-wise metrics for each fold
for fold in range(k): #k number of splits
    print(f"Fold {fold + 1}:")
    for class_label in range(number_of_classes):
        print(f'Class {class_label} - Precision: {precision_per_class[fold, class_label]:.3f}, '
              f'Recall: {recall_per_class[fold, class_label]:.3f}, '
              f'F1-score: {f1_per_class[fold, class_label]:.3f}')

# Calculate and print the macro average for each class
macro_precision_class = np.mean(precision_per_class, axis=0)
macro_recall_class = np.mean(recall_per_class, axis=0)
macro_f1_class = np.mean(f1_per_class, axis=0)

print("Macro Averages per class:")
for class_label in range(number_of_classes):
    print(f'Class {class_label} - Precision: {macro_precision_class[class_label]:.3f}, '
          f'Recall: {macro_recall_class[class_label]:.3f}, '
          f'F1-score: {macro_f1_class[class_label]:.3f}')

# # Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_fold, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate classification report 
class_rep = classification_report(y_test_fold, y_pred)

# Print classification report 
print("Classification Report:")
print(class_rep)
