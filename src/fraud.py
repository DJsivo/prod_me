import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import copy, hashlib, json, random, urllib.request
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
SEED = 42
class FraudMLP(nn.Module):
    def __init__(self, inputs=30, widths=(128,64), dropout=.2):
        super().__init__()
        layers=[]
        for width in widths:
            layers += [nn.Linear(inputs,width),nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(dropout)]
            inputs=width
        self.network=nn.Sequential(*layers,nn.Linear(inputs,1))
    def forward(self,x): return self.network(x).squeeze(1)

def load_data():
    path=ROOT/'data/raw/creditcard.csv'
    if not path.exists():
        temporary=path.with_suffix('.download')
        urllib.request.urlretrieve(SOURCE,temporary)
        temporary.replace(path)
    assert hashlib.md5(path.read_bytes()).hexdigest()=='e90efcb83d69faf99fcab8b0255024de', 'Контрольная сумма источника не совпала; проверьте файл'
    raw=pd.read_csv(path)
    assert raw.shape==(284807,31) and raw.Class.sum()==492
    assert list(raw.columns)==['Time']+[f'V{i}' for i in range(1,29)]+['Amount','Class']
    return raw

def predict(model,x,device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x,dtype=torch.float32,device=device))).cpu().numpy()

def threshold(y,p):
    precision,recall,t=precision_recall_curve(y,p)
    score=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    return float(t[np.argmax(score)])

def evaluate(y,p,t):
    pred=p>=t
    return dict(roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),f1=float(f1_score(y,pred)),accuracy=float(accuracy_score(y,pred)),threshold=t,confusion_matrix=confusion_matrix(y,pred).tolist())

def run():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'Для выполнения работы требуется доступная CUDA'
    device=torch.device('cuda')
    print('Устройство:', torch.cuda.get_device_name(0), flush=True)
    raw=load_data()
    # Точные дубликаты удаляются до разделения.
    data=raw.drop_duplicates().reset_index(drop=True)
    train, test=train_test_split(data,test_size=.2,stratify=data.Class,random_state=SEED)
    train, val=train_test_split(train,test_size=.25,stratify=train.Class,random_state=SEED)
    full_train=train
    # Сохраняем все мошенничества train и подвыборку обычных операций.
    train=pd.concat([train[train.Class==1],train[train.Class==0].sample(n=30000,random_state=SEED)]).sample(frac=1,random_state=SEED)
    features=data.columns.drop('Class').tolist()
    preprocessing=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    x=preprocessing.fit_transform(train[features]).astype('float32')
    xv=preprocessing.transform(val[features]).astype('float32')
    xt=preprocessing.transform(test[features]).astype('float32')
    y=train.Class.to_numpy(); yv=val.Class.to_numpy(); yt=test.Class.to_numpy()
    configs=[dict(widths=[128,64],dropout=.2,lr=.001),dict(widths=[64,32],dropout=.4,lr=.0005)]
    candidates=[]; histories=[]
    for config in configs:
        torch.manual_seed(SEED)
        model=FraudMLP(len(features),config['widths'],config['dropout']).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y==0).sum()/(y==1).sum()],device=device))
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.tensor(y,dtype=torch.float32)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(SEED),drop_last=True)
        best=-1; stale=0; history=[]
        for epoch in range(35):
            model.train(); losses=[]
            for bx,by in loader:
                optimizer.zero_grad(); loss=loss_fn(model(bx.to(device)),by.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
            ap=average_precision_score(yv,predict(model,xv,device))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_ap=float(ap)))
            if ap>best+1e-5: best=ap; state=copy.deepcopy(model.state_dict()); stale=0; best_epoch=epoch+1
            else: stale+=1
            if stale>=5: break
        model.load_state_dict(state)
        candidates.append((best,model,config,best_epoch)); histories.append(history)
        print('MLP',config,'лучшая эпоха',best_epoch,'AP на валидации',best,flush=True)
    best,model,config,best_epoch=max(candidates,key=lambda item:item[0])
    boost=LGBMClassifier(n_estimators=250,learning_rate=.05,num_leaves=15,class_weight='balanced',random_state=SEED,n_jobs=2,verbosity=-1)
    boost.fit(x,y)
    validation_predictions={'mlp':predict(model,xv,device),'lightgbm':boost.predict_proba(xv)[:,1]}
    test_predictions={'mlp':predict(model,xt,device),'lightgbm':boost.predict_proba(xt)[:,1]}
    metrics={name:evaluate(yt,p,threshold(yv,validation_predictions[name])) for name,p in test_predictions.items()}
    report=dict(source=SOURCE,sha256=hashlib.sha256((ROOT/'data/raw/creditcard.csv').read_bytes()).hexdigest(),raw_rows=len(raw),duplicates_removed=len(raw)-len(data),device=str(device),gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,split={name:dict(rows=len(frame),fraud=int(frame.Class.sum()),prevalence=float(frame.Class.mean())) for name,frame in [('train_full',full_train),('train_used',train),('validation',val),('test',test)]},selected_config=config,best_epoch=best_epoch,validation_ap={k:float(average_precision_score(yv,p)) for k,p in validation_predictions.items()},histories=histories,test=metrics)
    artifacts=ROOT/'models/fraud_bundle'; artifacts.mkdir(exist_ok=True)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},inputs=len(features),config=config,threshold=metrics['mlp']['threshold']),artifacts/'network.pt')
    joblib.dump(dict(preprocessing=preprocessing,features=features,baseline=boost),artifacts/'preprocessing.joblib')
    # Проверяем прогноз после загрузки сохранённых артефактов.
    loaded=torch.load(artifacts/'network.pt',weights_only=True,map_location='cpu')
    restored=FraudMLP(loaded['inputs'],loaded['config']['widths'],loaded['config']['dropout']); restored.load_state_dict(loaded['state_dict'])
    saved_preprocessing=joblib.load(artifacts/'preprocessing.joblib')
    restored_x=saved_preprocessing['preprocessing'].transform(test[saved_preprocessing['features']].iloc[:20]).astype('float32')
    np.testing.assert_allclose(predict(restored,restored_x,'cpu'),test_predictions['mlp'][:20],rtol=1e-4,atol=1e-6)
    (ROOT/'reports/fraud_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return data,preprocessing,features,report

def projections(data,preprocessing,features):
    import umap
    # Обогащённая выборка для наглядности: 2500 обычных и до 400 мошеннических операций.
    sample=pd.concat([data[data.Class==0].sample(n=2500,random_state=SEED),data[data.Class==1].sample(n=min(400,int(data.Class.sum())),random_state=SEED)])
    z=preprocessing.transform(sample[features]); y=sample.Class.to_numpy()
    pca=PCA(n_components=min(50,z.shape[1]),random_state=SEED); reduced=pca.fit_transform(z)
    coords={'PCA':reduced[:,:2],'t-SNE':TSNE(n_components=2,perplexity=30,random_state=SEED,init='pca',learning_rate='auto',max_iter=700).fit_transform(reduced),'UMAP':umap.UMAP(n_components=2,n_neighbors=20,random_state=SEED,n_jobs=1).fit_transform(reduced)}
    for name, xy in coords.items():
        fig,ax=plt.subplots(figsize=(8,5))
        for label,color in [(0,'steelblue'),(1,'darkorange')]:
            mask=y==label; ax.scatter(xy[mask,0],xy[mask,1],s=7,alpha=.5,c=color,label=f'Класс {label}: {mask.sum()}')
        ax.set(title=f'{name}: обогащённая выборка (не реальная доля классов)',xlabel='Компонента 1',ylabel='Компонента 2'); ax.legend(); fig.tight_layout(); plt.show()
    print('Компонент PCA:',reduced.shape[1],'; доля дисперсии первых двух:',pca.explained_variance_ratio_[:2].sum())
