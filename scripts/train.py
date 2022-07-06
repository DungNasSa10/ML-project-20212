# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models.ann import get_ann
from functional import plot_roc_curve
from pipelines import TFPipeline, SKPipeline

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
if model_type == 'ann':
    tp = TFPipeline(get_ann(), model_name=model_name)
    tp.train(X_train, y_train, epochs=param['epochs'], batch_size=param['batch_size'], verbose=1)
    tp.save_model()

    losses = pd.DataFrame(tp.get_history())
    losses[['loss']].plot(figsize=(14, 8))

    tp.test()

# =========================================================
elif model_type == 'rdf':
    sp = SKPipeline(RandomForestClassifier(n_estimators=param['n_estimators']), model_name=model_name)
    sp.traim(X_train, y_train)
    sp.save_model()
    sp.test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    plot_roc_curve(sp.model, X_test, y_test)
    
# =========================================================
elif model_type == "lgr":
    sp = SKPipeline(LogisticRegression(), model_name=model_name)
    sp.train(X_train=X_train, X_test=X_test)
    sp.save_model()
    sp.test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    plot_roc_curve(sp.model, X_test, y_test)

# =========================================================
# =========================================================
# =========================================================
# =========================================================