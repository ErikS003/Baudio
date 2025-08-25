import pickle

with open("bee_audio_rf.pkl", "rb") as f:
    data = pickle.load(f)

# If it’s a DataFrame
try:
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        print(data.head())   # preview first rows
except ImportError:
    pass

# If it’s a scikit-learn model
try:
    from sklearn import tree
    print(data)   # prints model info
except Exception:
    pass

import pickle

with open("bee_audio_rf.pkl", "rb") as f:
    obj = pickle.load(f)

model = obj["model"]

# General info
print(model)
print("Number of trees:", len(model.estimators_))

# Feature importances (which features matter most)
print("Feature importances:", model.feature_importances_)

# Look at one of the trees
from sklearn import tree
tree_i = model.estimators_[0]

# Text view of the first tree
print(tree.export_text(tree_i))

# Or visualize (needs matplotlib/graphviz)
import matplotlib.pyplot as plt
tree.plot_tree(tree_i, filled=True)
plt.show()
