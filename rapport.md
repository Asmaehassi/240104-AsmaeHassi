#  **Rapport Data Science — Détection du Risque de Maladie Cardiaque**

##  **1. Introduction**

La maladie cardiaque est l’une des premières causes de mortalité dans le monde. L’objectif de ce projet est de construire un modèle prédictif capable d’identifier si un patient présente un risque cardiaque à partir de données cliniques. Pour cela, nous utilisons le dataset **Heart Disease** disponible sur Kaggle.

Ce rapport détaille le cycle complet :

* Chargement et préparation des données
* Nettoyage et traitement
* Analyse Exploratoire (EDA)
* Modélisation Machine Learning (Random Forest)
* Évaluation des performances
* Interprétation logique et détaillée des résultats

---

##  **2. Chargement des Packages**

Nous avons utilisé les bibliothèques suivantes :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
```

Ces packages permettent :

* **Pandas / Numpy** → manipulation des données
* **Matplotlib / Seaborn** → visualisations
* **Scikit‑Learn** → préparation des données et Machine Learning
* **kagglehub** → téléchargement du dataset

---

##  **3. Chargement du Jeu de Données**

Le dataset est téléchargé automatiquement depuis Kaggle :

```python
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
df = pd.read_csv(path + "/heart.csv")
```

 **Dimensions du dataset :** (303 lignes, 14 colonnes)
 Chaque ligne représente un patient, avec des variables cliniques :

* âge
* cholestérol
* fréquence cardiaque maximale
* douleur thoracique
* pression artérielle
* électrocardiogramme
* etc.

La cible (*target*) vaut :

* **0** → pas de maladie
* **1** → maladie cardiaque

---

##  **4. Nettoyage et Préparation des Données**

Les données réelles contiennent souvent des valeurs manquantes. Pour simuler cela, nous avons ajouté 5% de NaN volontairement.

L’imputation est réalisée ainsi :

```python
imputer = SimpleImputer(strategy="mean")
X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
```

 -Remplacement des valeurs manquantes par la moyenne de chaque colonne

-Conservation des noms de colonnes

-Aucun NaN restant

---
