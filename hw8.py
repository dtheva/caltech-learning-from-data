import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import KFold

seed=0

def calc_error(y, h):
    return sum(y!=h)/len(y)

def check_one_vs_all():
    for t in [0,2,4,6,8,1,3,5,7,9]:
        df_tmp = df_train.copy()
        df_tmp["target"] = -1
        df_tmp.loc[df_tmp.digit==t,"target"] = 1
        
        model = SVC(C=.01,degree=2,kernel="poly",random_state=seed,coef0=1,gamma=1)    
        model.fit(df_tmp[predictors],df_tmp[target])
        df_tmp["prediction"] = model.predict(df_tmp[predictors])
        
        y = df_tmp.target
        h = df_tmp.prediction
        print("{} vs. all: {:.2f}".format(t, calc_error(y,h)))
        print("# of support vectors:",len(model.support_))

def filter_one_vs_one(df,x1,x2):
    df_out = df[df.digit.isin([x1,x2])].copy()
    df_out.digit.replace({x2:-1,x1:1},inplace=True)
    return df_out 

def check_one_vs_five():
    df_ovf = filter_one_vs_one(df_train,1,5)
    df_ovf_test = filter_one_vs_one(df_test,1,5)
    for C in [.0001,.001,.01,.1,1]:
        model = SVC(C=C,kernel="poly",degree=5,gamma=1,coef0=1,random_state=seed)
        model.fit(df_ovf[predictors],df_ovf.digit)
        df_ovf["prediction"] = model.predict(df_ovf[predictors])
        df_ovf_test["prediction"] = model.predict(df_ovf_test[predictors])
        
        print("C={} Error: {:.4f}".format(C,calc_error(df_ovf.digit,df_ovf.prediction)))
        print("C={} Error(out): {:.4f}".format(C,calc_error(df_ovf_test.digit,df_ovf_test.prediction)))
        print("# of support vectors:",len(model.support_))
              
def check_cv():
    df_cv = filter_one_vs_one(df_train,1,5)
    model = SVC(kernel="poly",degree=2,gamma=1,coef0=1)
    params = {
            "C": [.0001,.001,.01,.1,1]
            }
    scorer = make_scorer(calc_error,greater_is_better=False)
    
    scores = {}
    scores_list = {}
    for i in range(100):
        cv = KFold(n_splits=10,shuffle=True,random_state=i)
        gs = GridSearchCV(model,params,scoring=scorer,cv=cv)
        gs.fit(df_cv[predictors],df_cv.digit)
        
        best_param = gs.best_params_["C"]
        best_score = gs.best_score_
        if best_param not in scores:
            scores[best_param] = 1
        else:
            scores[best_param] += 1
        if best_param not in scores_list:
            scores_list[best_param] = []
        else:
            scores_list[best_param].append(best_score)
    print("Score count:")
    for k, v in scores.items():
        print("C={}: {} with average score {:.3f}".format(k, v, pd.np.abs(pd.np.mean(scores_list[k]))))
    
def check_rbf():
    df_rbf = filter_one_vs_one(df_train,1,5)
    df_rbf_test = filter_one_vs_one(df_test,1,5)
    model = SVC(kernel="rbf",gamma=1,coef0=1)
    params = {
            "C": [.01,1,100,10**4,10**6]
            }
    scorer = make_scorer(calc_error,greater_is_better=False)
    gs = GridSearchCV(model,params,scoring=scorer,cv=10)
    gs.fit(df_rbf[predictors],df_rbf.digit)
    
    print("Best params:", gs.best_params_)

    for C in params["C"]:
        model = SVC(C=C,kernel="rbf",gamma=1,coef0=1)
        model.fit(df_rbf[predictors],df_rbf.digit)
        df_rbf["prediction"] = model.predict(df_rbf[predictors])
        df_rbf_test["prediction"] = model.predict(df_rbf_test[predictors])
        
        print("E(in) for C={}: {:.3f}".format(C, calc_error(df_rbf.digit, df_rbf.prediction)))
        print("E(out) for C={}: {:.3f}".format(C, calc_error(df_rbf_test.digit,df_rbf_test.prediction)))
    
df_train = pd.read_csv("features.train",delim_whitespace=True,header=None)
df_test = pd.read_csv("features.test",delim_whitespace=True,header=None)

cols = ["digit", "intensity", "symmetry"]
df_train.columns = cols
df_test.columns = cols

target = "target"
predictors = ["intensity","symmetry"]

check_one_vs_all()
check_one_vs_five()
check_cv()
check_rbf()