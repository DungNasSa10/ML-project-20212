# %%
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models import get_ann
from functional import print_score, roc_auc_score, f1_score, plot_roc_curve

# =========================================================
# %% Load args
import sys
import os
from args import get_parser, parse_args_to_dict

parser      = get_parser()

try: 
    args    = parser.parse_args(sys.argv[1:])
except:
    args    = parser.parse_args('')

param       = parse_args_to_dict(args)
print('param =', param)

# =========================================================
# %%
model_name = param['name']
model_type = param['model']

checkpoint_path = f"checkpoints/{model_name}/"
if os.path.exists(checkpoint_path) is False:
    os.makedirs(checkpoint_path)

# =========================================================
# %% Load elite data
accepted_loans = pd.read_csv(param['data'])

# =========================================================
# %%
features = accepted_loans.loc[:, accepted_loans.columns != 'loan_paid'].values
labels = accepted_loans['loan_paid'].values

# =========================================================
# %% 
print("features.shape =", features.shape)
print("labels.shape   =", labels.shape)

# =========================================================
# %% Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# =========================================================
# %% Scale data
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# %%
print("X_train.shape =", X_train.shape)
print("X_test.shape  =", X_test.shape)

# =========================================================
# %%
model_save_path = checkpoint_path + 'model.pkl'

def print_info(info: dict):
    print()
    for k, v in info.items():
        print(k, "=")
        print(v)
    print()

if model_type == 'ann':
    model = get_ann()

    model.fit(x=X_train, 
            y=y_train, 
            epochs=param['epochs'],
            batch_size=param['batch_size'],
            verbose=1)

    losses = pd.DataFrame(model.history.history)
    losses[['loss']].plot(figsize=(14, 8))
    
    history_save_path = checkpoint_path + 'history.pkl'
    with open(history_save_path, "wb") as f:
        pkl.dump(losses, f)

    predictions = (model.predict(X_test) > 0.5).astype("int32")

    info = {
        "test_score" : print_score(y_test, predictions, train=False),
        "f1" : f1_score(predictions, y_test),
        "roc_auc_score" : roc_auc_score(predictions, y_test)
    }

    print_info(info)

    model.save(checkpoint_path + "model")
    print("Save model at:", checkpoint_path + "model")

# =========================================================
elif model_type == 'rdf':
    model = RandomForestClassifier(n_estimators=param['n_estimators'])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    info = {
        "train_score" : print_score(y_train, y_train_pred, train=True),
        "test_score" : print_score(y_test, y_test_pred, train=False),
        "f1" : f1_score(y_test_pred, y_test),
    }

    print_info(info)

    plot_roc_curve(model, X_test, y_test)
    
    with open(model_save_path, 'wb') as f:
        pkl.dump(model, f)
    print("Save model at:", model_save_path)

# =========================================================
elif model_type == "lgr":
    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    info = {
        "train_score" : print_score(y_train, y_train_pred, train=True),
        "test_score" : print_score(y_test, y_test_pred, train=False),
        "f1" : f1_score(y_test_pred, y_test),
        "roc_auc_score" : roc_auc_score(y_test_pred, y_test)
    }
    
    print_info(info)

    plot_roc_curve(model, X_test, y_test)

    with open(model_save_path, 'wb') as f:
        pkl.dump(model, f)
    print("Save model at:", model_save_path)
    
# =========================================================
# %% Save info
for k in param.keys():
    info[k] = param[k]

info_save_path = checkpoint_path + "info.pkl"
with open(info_save_path, 'wb') as f:
    pkl.dump(model, f)
print("Save information at:", info_save_path)

# =========================================================
# =========================================================
# =========================================================