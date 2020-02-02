# %%
"""
## IFT870 - TP1 
### gibg2501 - leba3207
"""

# %%
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
import pandas as pd
%matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns

# removes maximum number of columns & rows for display
pd.options.display.max_columns = None
pd.options.display.max_rows = None

tp1_data_file = 'TP1_data.csv'
tp1_data = pd.read_csv(tp1_data_file, header=0, index_col=0)
tp1_data

# %%
"""
Description des données statistiques de base du jeux de données
"""

# %%
tp1_data.describe()

# %%
"""
Affichage des attributs du jeux de données paire à paire
"""

# %%
%matplotlib inline
sns.pairplot(tp1_data, vars=['attribut1', 'attribut2', 'attribut3', 'attribut4'],  hue='classe')