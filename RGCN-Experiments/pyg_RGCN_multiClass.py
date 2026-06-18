import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE


rel_no=70
path="/home/faisopos/workspace/marios/RGCN/"

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values.astype(str), show_progress_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()


class TypesEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def normaliseRelation(rel):
    
    if rel=="ADMINISTERED_TO":
        norm_rel="ADMINISTERED_TO"
    elif rel=="ADMINISTERED_TO__SPEC__":
        norm_rel= "ADMINISTERED_TO"
    elif rel=="AFFECTS":
        norm_rel= "AFFECTS"        
    elif rel=="AFFECTS__SPEC__":
        norm_rel= "AFFECTS"
    elif rel=="ASSOCIATED_WITH":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__INFER__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="ASSOCIATED_WITH__SPEC__":
        norm_rel= "ASSOCIATED_WITH"
    elif rel=="AUGMENTS":
        norm_rel= "AUGMENTS"
    elif rel=="AUGMENTS__SPEC__":
        norm_rel= "AUGMENTS"
    elif rel=="CAUSES":
        norm_rel= "CAUSES"
    elif rel=="CAUSES__SPEC__":
        norm_rel= "CAUSES"
    elif rel=="COEXISTS_WITH":
        norm_rel= "COEXISTS_WITH"
    elif rel=="COEXISTS_WITH__SPEC__":
        norm_rel= "COEXISTS_WITH"
    elif rel=="compared_with":
        norm_rel= "compared_with"
    elif rel=="compared_with__SPEC__":
        norm_rel= "compared_with"
    elif rel=="COMPLICATES":
        norm_rel= "COMPLICATES"
    elif rel=="COMPLICATES__SPEC__":
        norm_rel= "COMPLICATES"
    elif rel=="CONVERTS_TO":
        norm_rel= "CONVERTS_TO"
    elif rel=="CONVERTS_TO__SPEC__":
        norm_rel= "CONVERTS_TO"
    elif rel=="DIAGNOSES":
        norm_rel= "DIAGNOSES"
    elif rel=="DIAGNOSES__SPEC__":
        norm_rel= "DIAGNOSES"
    elif rel=="different_from":
        norm_rel= "different_from"
    elif rel=="different_from__SPEC__":
        norm_rel= "different_from"
    elif rel=="different_than":
        norm_rel= "different_than"
    elif rel=="different_than__SPEC__":
        norm_rel= "different_than"
    elif rel=="DISRUPTS":
        norm_rel= "DISRUPTS"
    elif rel=="DISRUPTS__SPEC__":
        norm_rel= "DISRUPTS"
    elif rel=="higher_than":
        norm_rel= "higher_than"
    elif rel=="higher_than__SPEC__":
        norm_rel= "higher_than"
    elif rel=="INHIBITS":
        norm_rel= "INHIBITS"
    elif rel=="INHIBITS__SPEC__":
        norm_rel= "INHIBITS"
    elif rel=="INTERACTS_WITH":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__INFER__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="INTERACTS_WITH__SPEC__":
        norm_rel= "INTERACTS_WITH"
    elif rel=="IS_A":
        norm_rel= "IS_A"
    elif rel=="ISA":
        norm_rel= "ISA"
    elif rel=="LOCATION_OF":
        norm_rel= "LOCATION_OF"
    elif rel=="LOCATION_OF__SPEC__":
        norm_rel= "LOCATION_OF"
    elif rel=="lower_than":
        norm_rel= "lower_than"
    elif rel=="lower_than__SPEC__":
        norm_rel= "lower_than"
    elif rel=="MANIFESTATION_OF":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="MANIFESTATION_OF__SPEC__":
        norm_rel= "MANIFESTATION_OF"
    elif rel=="METHOD_OF":
        norm_rel= "METHOD_OF"
    elif rel=="METHOD_OF__SPEC__":
        norm_rel= "METHOD_OF"
    elif rel=="OCCURS_IN":
        norm_rel= "OCCURS_IN"
    elif rel=="OCCURS_IN__SPEC__":
        norm_rel= "OCCURS_IN"
    elif rel=="PART_OF":
        norm_rel= "PART_OF"
    elif rel=="PART_OF__SPEC__":
        norm_rel= "PART_OF"
    elif rel=="PRECEDES":
        norm_rel= "PRECEDES"
    elif rel=="PRECEDES__SPEC__":
        norm_rel= "PRECEDES"
    elif rel=="PREDISPOSES":
        norm_rel= "PREDISPOSES"
    elif rel=="PREDISPOSES__SPEC__":
        norm_rel= "PREDISPOSES"
    elif rel=="PREVENTS":
        norm_rel= "PREVENTS"
    elif rel=="PREVENTS__SPEC__":
        norm_rel= "PREVENTS"
    elif rel=="PROCESS_OF":
        norm_rel= "PROCESS_OF"
    elif rel=="PROCESS_OF__SPEC__":
        norm_rel= "PROCESS_OF"
    elif rel=="PRODUCES":
        norm_rel= "PRODUCES"
    elif rel=="PRODUCES__SPEC__":
        norm_rel= "PRODUCES"
    elif rel=="same_as":
        norm_rel= "same_as"
    elif rel=="same_as__SPEC__":
        norm_rel= "same_as"
    elif rel=="STIMULATES":
        norm_rel= "STIMULATES"
    elif rel=="STIMULATES__SPEC__":
        norm_rel= "STIMULATES"
    elif rel=="IS_TREATED":
        norm_rel= "IS_TREATED"
    elif rel=="USES":
        norm_rel= "USES"
    elif rel=="USES__SPEC__":
        norm_rel= "USES"
    elif rel=="MENTIONED_IN":
        norm_rel= "MENTIONED_IN"
    elif rel=="HAS_MESH":
        norm_rel= "HAS_MESH"
    else:
        norm_rel="ASSOCIATED_WITH"

    return norm_rel

def encodeEdgeTypes(df):
    reltypes = ["ADMINISTERED_TO","AFFECTS","ASSOCIATED_WITH","AUGMENTS","CAUSES","COEXISTS_WITH","compared_with","COMPLICATES","CONVERTS_TO","DIAGNOSES","different_from","different_than","DISRUPTS","higher_than","INHIBITS","INTERACTS_WITH","IS_A","ISA","LOCATION_OF","lower_than","MANIFESTATION_OF","METHOD_OF","OCCURS_IN","PART_OF","PRECEDES","PREDISPOSES","PREVENTS","PROCESS_OF","PRODUCES","same_as","STIMULATES","IS_TREATED","USES","MENTIONED_IN","HAS_MESH"]
    mapping = {rtype: i for i, rtype in enumerate(reltypes)}
    x = torch.zeros(len(df), dtype=torch.int64)
    for i, col in enumerate(df.values):
        rel=normaliseRelation(col)
#        print('edgetype i, x[i]: ',i, type(mapping[rel]))
        x[i]= mapping[rel]
    return x



def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, sep='\t', **kwargs)

    mapping = {index: i for i, index in enumerate(df.index.unique())}
    
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, sep='\t', **kwargs)
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
#        print('edge_attrs: ', edge_attrs)
        edge_attr = torch.cat(edge_attrs, dim=-1)
#        print('CONCATENATED edge_attr: ', edge_attr)
    return edge_index, edge_attr


################################################
###################Load CSV#####################
################################################

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#print(torch.version.cuda)

ent_file_path=path+"entities.tsv"
rel_file_path=path+"relations.tsv"
rel1_file_path=path+"relations_entities.tsv"
rel2_file_path=path+"relations_articles.tsv"
art_file_path=path+"articles.tsv"


entity_cui, entity_type = load_node_csv(
    ent_file_path, index_col='ID', encoders={
        'CUI': IdentityEncoder(dtype=torch.long),
        'SEM_TYPES': TypesEncoder()
    })

article_title, article_no = load_node_csv(
    art_file_path, index_col='AID', encoders={
        'TITLE': SequenceEncoder(),
        'ARTICLE_NO': IdentityEncoder(dtype=torch.int)
    })

from torch_geometric.data import HeteroData


data = HeteroData()

data['entity'].x = entity_cui
data['article'].x = article_title

#Insert and Encode relations

print('load entity-entity rels')

edge_index, edge_attr = load_edge_csv(
    rel1_file_path,
    src_index_col='NOD1',
    src_mapping=entity_type,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)#,  'RELATION': InteractsEncoder() 
},
)

data['entity', 'rel', 'entity'].edge_index = edge_index
data['entity', 'rel', 'entity'].edge_attr = edge_attr

print('edge_index=', edge_index)

print('load article-entity rels')

article_no_int = {}
for key, value in article_no.items():
    try:
        article_no_int[int(float(key))] = value
    except ValueError:
        print("Not a float"+ " "+ str(key)+" "+str(value))

edge_index, edge_attr = load_edge_csv(
    rel2_file_path,
    src_index_col='NOD1',
    src_mapping=article_no_int,
    dst_index_col='NOD2',
    dst_mapping=entity_type,
    encoders={'REFERENCES': IdentityEncoder(dtype=torch.long)
#          'RELATION': InteractsEncoder()
    },
)

data['article', 'rel', 'entity'].edge_index = edge_index
data['article', 'rel', 'entity'].edge_attr = edge_attr

df = pd.read_csv(rel1_file_path, sep='\t')
df_rel=df["RELATION"]
edge_type = encodeEdgeTypes(df_rel)


################################################
###################GCN##########################
################################################


""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.
Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""

from torch.nn import Parameter
from tqdm import tqdm

from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


###TEST Multi-layer Perceptron (MLP) decoder over concatenated node embeddings
class MultiClassLinkDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z, edge_index, class_no):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        edge_repr = torch.cat([z_src, z_dst], dim=1)  # (batch_size, 2*hidden_dim)
        ###up sample pos classes x2
        if class_no==10: # if class_no!=0: 
            artif_edge_repr = torch.cat((edge_repr, edge_repr), dim=0)
            artif_edge_repr = torch.cat((artif_edge_repr, edge_repr), dim=0)
            intented_size = 2*edge_repr.size(dim=0)
            y_needed = torch.cat([torch.ones(edge_repr.size(dim=0)), torch.zeros(intented_size)])
            smote = SMOTE(random_state=42)
            artif_edge_repr,y = smote.fit_resample(artif_edge_repr.detach(), y_needed.detach())
            l= artif_edge_repr.shape[0]
            edge_repr=torch.from_numpy(artif_edge_repr[int(l//2):l,:])
        ###up sample pos classes x3
        if class_no!=0: 
            artif_edge_repr = torch.cat((edge_repr, edge_repr), dim=0)
            artif_edge_repr = torch.cat((artif_edge_repr, edge_repr), dim=0)
            artif_edge_repr = torch.cat((artif_edge_repr, edge_repr), dim=0)
            intented_size = 3*edge_repr.size(dim=0)
            y_needed = torch.cat([torch.ones(edge_repr.size(dim=0)), torch.zeros(intented_size)])
            smote = SMOTE(random_state=42)
            artif_edge_repr,y = smote.fit_resample(artif_edge_repr.detach(), y_needed.detach())
            l= artif_edge_repr.shape[0]
            edge_repr=torch.from_numpy(artif_edge_repr[int(l//2):l,:])
            
        logits = self.mlp(edge_repr)  # (batch_size, num_classes)
        return logits  # raw scores (to use with CrossEntropyLoss)


model = GAE(
    RGCNEncoder(data.num_nodes, hidden_channels=300,
                num_relations=rel_no),
    MultiClassLinkDecoder(hidden_dim=300, num_classes=6),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    #print('Now training the RGCN model:')
    model.train()
    optimizer.zero_grad()

    #print('RGCNEncoder running...')  
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)

    #print('Decoder running...')
    
    #neg_edge_index = negative_sampling(train_edge_index, data.num_nodes)
    #neg_out = model.decode(z, neg_edge_index, train_edge_type)
    neg_out = model.decode(z, orig_neg_edge_index, 0)#, orig_neg_edge_type)

    pos_class1_out = model.decode(z, train_class1_edge_index, 1)#, train_class1_edge_type)
    pos_class2_out = model.decode(z, train_class2_edge_index, 2)#, train_class2_edge_type)
    pos_class3_out = model.decode(z, train_class3_edge_index, 3)#, train_class3_edge_type)
    pos_class4_out = model.decode(z, train_class4_edge_index, 4)#, train_class4_edge_type)
    pos_class5_out = model.decode(z, train_class5_edge_index, 5)#, train_class5_edge_type)

    out = torch.cat([neg_out, pos_class1_out, pos_class2_out,pos_class3_out, pos_class4_out, pos_class5_out])
    true_labels = torch.cat([torch.zeros_like(neg_out), torch.ones_like(pos_class1_out), torch.full_like(pos_class2_out, 2), torch.full_like(pos_class3_out, 3), torch.full_like(pos_class4_out, 4), torch.full_like(pos_class5_out, 5)])

    loss = torch.nn.CrossEntropyLoss()
    output = loss(out, true_labels)#.squeeze().long())
    output.backward()

    optimizer.step()

    return float(output)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data['entity', 'rel', 'entity'].edge_index, edge_type)

    tps=[0,0,0,0,0,0]
    fps=[0,0,0,0,0,0]
    fns=[0,0,0,0,0,0]
    rec=[0.0,0.0,0.0,0.0,0.0,0.0]
    prec=[0.0,0.0,0.0,0.0,0.0,0.0]
    f1=[0.0,0.0,0.0,0.0,0.0,0.0]
    
    tps, fps, fns = calculate_hits(z, test_neg_edge_index, 0, tps, fps, fns)
    tps, fps, fns = calculate_hits(z, test_class1_edge_index, 1, tps, fps, fns)
    ##DEBUG
    #print('@@@@@@@@@@@@test_class1_edge_index[0][0], [0][1]=', test_class1_edge_index[0][0], ',', test_class1_edge_index[0][1], '  label=1')

    tps, fps, fns = calculate_hits(z, test_class2_edge_index, 2, tps, fps, fns)
    tps, fps, fns = calculate_hits(z, test_class3_edge_index, 3, tps, fps, fns)
    tps, fps, fns = calculate_hits(z, test_class4_edge_index, 4, tps, fps, fns)
    tps, fps, fns = calculate_hits(z, test_class5_edge_index, 5, tps, fps, fns)

    ###Now calculate and print class performance metrics:
    for i in range(6):
        if not (tps[i]==0 and fps[i]==0):
            prec[i]=float(tps[i])/float(tps[i]+fps[i])
        if not (tps[i]==0 and fns[i]==0):
            rec[i]=float(tps[i])/float(tps[i]+fns[i])
        if not (prec[i]==0.0 and rec[i]==0.0):
            f1[i]=2*prec[i]*rec[i]/(prec[i]+rec[i])
        #print('For Class ', i, ' precision=', prec[i], ' recall=', rec[i], ' f1-score=',f1[i])
    ###calculate average
    #precision = sum(prec)/6.0
    #recall = sum(rec)/6.0
    #f1_score = sum(f1)/6.0
    return prec, rec, f1


@torch.no_grad()
def calculate_hits (z, eval_edge_index, pol, tps, fps, fns):
    out = model.decode(z, eval_edge_index, pol)
    for class_scores in out:
        max_score=-10000.8
        max_index=-1
        for i in range(6):
            if (class_scores[i].item())>max_score:
                max_score=class_scores[i].item()
                max_index=i
        
        if max_index==pol:
            tps[pol]=tps[pol]+1
        else:
            fps[max_index]=fps[max_index]+1
            fns[pol]=fns[pol]+1
    return tps, fps, fns



################################################
################### Groundtruth ################
################################################

groundtruth=path+"posGroundtruth_filtered.tsv"
neg_groundtruth=path+"negGroundtruth_filtered.tsv"

print('load entity-entity TRAIN rels')

ie = IdentityEncoder(dtype=torch.long)

df_ground= pd.read_csv(groundtruth, sep='\t')
df_neg_ground= pd.read_csv(neg_groundtruth, sep='\t')
neg_rtypes = df_neg_ground["RELATION"]
gener_refs = [0]*len(neg_rtypes)
df_neg_ground['REFERENCES'] = gener_refs
 
df = pd.concat([df_ground, df_neg_ground])

#dfLabels=["NOD1","NOD2","RELATION"]
df_pairs_rtypes= df.drop('REFERENCES', axis=1)
refs = df["REFERENCES"]

####undersampling original groundtruth ratio
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print('Initial samples distribution: ',Counter(refs))


################################################
################### k-Fold-CV ##################
################################################
from sklearn.model_selection import StratifiedKFold
k=10
i=0

FNs=0
TPs=0
FPs=0

VALIDATION_FSCORE10= []
VALIDATION_FSCORE5= []
VALIDATION_FSCORE15= []
VALIDATION_FSCORE20= []
VALIDATION_PRECISION = []
VALIDATION_RECALL= []
VALIDATION_FSCORE= []

rows, cols = (6, 10)
CLASS_FSCORE = [ [ None for i in range(cols) ] for j in range(rows) ]


skf = StratifiedKFold(n_splits = k, random_state = 1, shuffle = True)
for train_index, test_index in skf.split(df_pairs_rtypes, refs):
    i=i+1
    print('Fold ', i, ' - start training...')
    df_train_pairs_rtypes=df_pairs_rtypes.iloc[train_index]
    df_train_refs=refs.iloc[train_index]
#    print('df_train_pairs len: ', len(df_train_pairs_rtypes))
#    print('df_train_pairs: ', df_train_pairs_rtypes)
#    print('df_train_ref len:', len (df_train_refs))
#    print('df_train_ref :', df_train_refs)
    df_train=pd.concat([df_train_pairs_rtypes, df_train_refs], axis=1)
#    print('df_train len: ', len(df_train))
#    print('df_train: ', df_train)


    #remove neg records from training...
    df_neg_train = df_train[df_train['REFERENCES'] == 0]
    ####Downsampling negatives to 1/100
    df_neg_train = df_neg_train.sample(frac=0.01, axis=0)
    df_class1_train = df_train[df_train['REFERENCES'] == 1000]
    df_class2_train = df_train[df_train['REFERENCES'] == 2000]
    df_class3_train = df_train[df_train['REFERENCES'] == 3000]
    df_class4_train = df_train[df_train['REFERENCES'] == 4000]
    df_class5_train = df_train[df_train['REFERENCES'] == 5000]

    df_test_refs=refs.iloc[test_index]
    df_test_pairs_rtypes=df_pairs_rtypes.iloc[test_index]
    df_test=pd.concat([df_test_pairs_rtypes, df_test_refs], axis=1)
    all_t_test = df_test["RELATION"]
    src0 = [entity_type[index] for index in df_test["NOD1"]]
    dst0 = [entity_type[index] for index in df_test["NOD2"]]
    all_test_edge_index = torch.tensor([src0, dst0])
    all_test_edge_type = encodeEdgeTypes(all_t_test)
    print("Test samples distribution: " , Counter(df_test["REFERENCES"]))
    df_neg_test = df_test[df_test['REFERENCES'] == 0] 
    df_class1_test = df_test[df_test['REFERENCES'] == 1000]
    df_class2_test = df_test[df_test['REFERENCES'] == 2000]
    df_class3_test = df_test[df_test['REFERENCES'] == 3000]
    df_class4_test = df_test[df_test['REFERENCES'] == 4000]
    df_class5_test = df_test[df_test['REFERENCES'] == 5000]   


    t_1_train = df_class1_train["RELATION"]
    pairLabels=["NOD1","NOD2"]
    p_1_train = df_class1_train[pairLabels]
    a_1_train=df_class1_train["REFERENCES"]
    src1 = [entity_type[index] for index in p_1_train["NOD1"]]
    dst1 = [entity_type[index] for index in p_1_train["NOD2"]]
    train_class1_edge_index = torch.tensor([src1, dst1])
    train_class1_edge_type = encodeEdgeTypes(t_1_train)
    train_class1_edge_attrs = [ie(a_1_train)]
    train_class1_edge_attr = torch.cat(train_class1_edge_attrs, dim=-1)

    t_2_train = df_class2_train["RELATION"]
    p_2_train = df_class2_train[pairLabels]
    a_2_train=df_class2_train["REFERENCES"]
    src2 = [entity_type[index] for index in p_2_train["NOD1"]]
    dst2 = [entity_type[index] for index in p_2_train["NOD2"]]
    train_class2_edge_index = torch.tensor([src2, dst2])
    train_class2_edge_type = encodeEdgeTypes(t_2_train)
    train_class2_edge_attrs = [ie(a_2_train)]
    train_class2_edge_attr = torch.cat(train_class2_edge_attrs, dim=-1)

    t_3_train = df_class3_train["RELATION"]
    p_3_train = df_class3_train[pairLabels]
    a_3_train=df_class3_train["REFERENCES"]
    src3 = [entity_type[index] for index in p_3_train["NOD1"]]
    dst3 = [entity_type[index] for index in p_3_train["NOD2"]]
    train_class3_edge_index = torch.tensor([src3, dst3])
    train_class3_edge_type = encodeEdgeTypes(t_3_train)
    train_class3_edge_attrs = [ie(a_3_train)]
    train_class3_edge_attr = torch.cat(train_class3_edge_attrs, dim=-1)

    t_4_train = df_class4_train["RELATION"]
    p_4_train = df_class4_train[pairLabels]
    a_4_train=df_class4_train["REFERENCES"]
    src4 = [entity_type[index] for index in p_4_train["NOD1"]]
    dst4 = [entity_type[index] for index in p_4_train["NOD2"]]
    train_class4_edge_index = torch.tensor([src4, dst4])
    train_class4_edge_type = encodeEdgeTypes(t_4_train)
    train_class4_edge_attrs = [ie(a_4_train)]
    train_class4_edge_attr = torch.cat(train_class4_edge_attrs, dim=-1)

    t_5_train = df_class5_train["RELATION"]
    p_5_train = df_class5_train[pairLabels]
    a_5_train=df_class5_train["REFERENCES"]
    src5 = [entity_type[index] for index in p_5_train["NOD1"]]
    dst5 = [entity_type[index] for index in p_5_train["NOD2"]]
    train_class5_edge_index = torch.tensor([src5, dst5])
    train_class5_edge_type = encodeEdgeTypes(t_5_train)
    train_class5_edge_attrs = [ie(a_5_train)]
    train_class5_edge_attr = torch.cat(train_class5_edge_attrs, dim=-1)
####TRAIN WITH Groundtruth negative sample!
    neg_p_train = df_neg_train[pairLabels]
    nsrc = [entity_type[index] for index in neg_p_train["NOD1"]]
    ndst = [entity_type[index] for index in neg_p_train["NOD2"]]
    orig_neg_edge_index = torch.tensor([nsrc, ndst])
    neg_t_train = df_neg_train["RELATION"]
    orig_neg_edge_type = encodeEdgeTypes(neg_t_train)
####



##    p_train=pairs[train_index]
##    p_test=pairs[test_index]
##    t_train=rtypes[train_index]
##    t_test=rtypes[test_index]
##    a_train=refs[train_index]
##    a_test=refs[test_index]

    t_1_test = df_class1_test["RELATION"]
    p_1_test = df_class1_test[pairLabels]
    a_1_test=df_class1_test["REFERENCES"]
    src1 = [entity_type[index] for index in p_1_test["NOD1"]]
    dst1 = [entity_type[index] for index in p_1_test["NOD2"]]
    test_class1_edge_index = torch.tensor([src1, dst1])
    #DEBUG
    print('p_1_test', p_1_test.iloc[0])
    print('test_class1_edge_index[0][0],[0][1]', test_class1_edge_index[0][0], ' ,', test_class1_edge_index[0][1])
    test_class1_edge_type = encodeEdgeTypes(t_1_test)
    test_class1_edge_attrs = [ie(a_1_test)]
    test_class1_edge_attr = torch.cat(test_class1_edge_attrs, dim=-1)

    t_2_test = df_class2_test["RELATION"]
    p_2_test = df_class2_test[pairLabels]
    a_2_test=df_class2_test["REFERENCES"]
    src2 = [entity_type[index] for index in p_2_test["NOD1"]]
    dst2 = [entity_type[index] for index in p_2_test["NOD2"]]
    test_class2_edge_index = torch.tensor([src2, dst2])
    test_class2_edge_type = encodeEdgeTypes(t_2_test)
    test_class2_edge_attrs = [ie(a_2_test)]
    test_class2_edge_attr = torch.cat(test_class2_edge_attrs, dim=-1)

    t_3_test = df_class3_test["RELATION"]
    p_3_test = df_class3_test[pairLabels]
    a_3_test=df_class3_test["REFERENCES"]
    src3 = [entity_type[index] for index in p_3_test["NOD1"]]
    dst3 = [entity_type[index] for index in p_3_test["NOD2"]]
    test_class3_edge_index = torch.tensor([src3, dst3])
    test_class3_edge_type = encodeEdgeTypes(t_3_test)
    test_class3_edge_attrs = [ie(a_3_test)]
    test_class3_edge_attr = torch.cat(test_class3_edge_attrs, dim=-1)

    t_4_test = df_class4_test["RELATION"]
    p_4_test = df_class4_test[pairLabels]
    a_4_test=df_class4_test["REFERENCES"]
    src4 = [entity_type[index] for index in p_4_test["NOD1"]]
    dst4 = [entity_type[index] for index in p_4_test["NOD2"]]
    test_class4_edge_index = torch.tensor([src4, dst4])
    test_class4_edge_type = encodeEdgeTypes(t_4_test)
    test_class4_edge_attrs = [ie(a_4_test)]
    test_class4_edge_attr = torch.cat(test_class4_edge_attrs, dim=-1)

    t_5_test = df_class5_test["RELATION"]
    p_5_test = df_class5_test[pairLabels]
    a_5_test=df_class5_test["REFERENCES"]
    src5 = [entity_type[index] for index in p_5_test["NOD1"]]
    dst5 = [entity_type[index] for index in p_5_test["NOD2"]]
    test_class5_edge_index = torch.tensor([src5, dst5])
    test_class5_edge_type = encodeEdgeTypes(t_5_test)
    test_class5_edge_attrs = [ie(a_5_test)]
    test_class5_edge_attr = torch.cat(test_class5_edge_attrs, dim=-1)

    neg_t_test = df_neg_test["RELATION"]
    neg_p_test = df_neg_test[pairLabels]
    neg_a_test=df_neg_test["REFERENCES"]
    src0 = [entity_type[index] for index in neg_p_test["NOD1"]]
    dst0 = [entity_type[index] for index in neg_p_test["NOD2"]]
    test_neg_edge_index = torch.tensor([src0, dst0])
    test_neg_edge_type = encodeEdgeTypes(neg_t_test)
    ####

##FOR DEBUG PURPOSEs
    prec=0.0
    rec=0.0
    f1=0.0
    #model = GAE(RGCNEncoder(data.num_nodes, hidden_channels=100, num_relations=rel_no),DistMultDecoder(rel_no // 2 , hidden_channels=100),)
    model = GAE(RGCNEncoder(data.num_nodes, hidden_channels=300, num_relations=rel_no),MultiClassLinkDecoder(hidden_dim=300, num_classes=6),)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ####Training....
    TP=0
    FP=0
    FN=0
    f1_score=[]
    for epoch in range(1, 101):
        loss = train()
        if (epoch % 5) == 0:
            print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
            precision, recall, f1_score = test()
            prec = sum(precision)/6.0
            rec = sum(recall)/6.0
            f1 = sum(f1_score)/6.0

            for j in range(6):
                print('For Class ', j, ' precision=', precision[j], ' recall=', recall[j], ' f1-score=',f1_score[j])
            print('Fold ', i, ', Epoch: ', epoch, '. Prec: ', prec, ' recall: ', rec, 'f1: ', f1)
            
            if epoch==5:
                VALIDATION_FSCORE5.append(f1)
            if epoch==10:
                VALIDATION_FSCORE10.append(f1)
            if epoch==15:
                VALIDATION_FSCORE15.append(f1)
            if epoch==20:
                VALIDATION_FSCORE20.append(f1)

    VALIDATION_PRECISION.append(prec)
    VALIDATION_RECALL.append(rec)
    VALIDATION_FSCORE.append(f1)

    print('save class f1-scores: ', f1_score) 
    for j in range(6):
        CLASS_FSCORE[j][i-1]=f1_score[j]
    print('CLASS_FSCORE: ', CLASS_FSCORE)
    
###################################################
##########Print marco-average of metric values#####
###################################################
from numpy import mean             
print("Average F1 after 5 epochs: ", mean(VALIDATION_FSCORE5))
print("Average F1 after 10 epochs: ", mean(VALIDATION_FSCORE10))
#print("Average F1 after 15 epochs: ", mean(VALIDATION_FSCORE15))
#print("Average F1 after 30 epochs: ", mean(VALIDATION_FSCORE20))

#print('CLASS_FSCORE: ', CLASS_FSCORE)
for j in range(6):
    f1=0.0
    for i in range(10):
        f1=f1+CLASS_FSCORE[j][i]
    f1=f1/10.0
    print('For Class ', j, ' Average f1-score after ALL epochs is: f1=',f1)

print("\nTest Set Metrics of the trained model:")
print("Average Precision: ", mean(VALIDATION_PRECISION))
print("Average Recall: ", mean(VALIDATION_RECALL))
print("Average F1-score: ", mean(VALIDATION_FSCORE))



