#  Heart Disease Prediction : Projet Data Science

##  Description
Projet de prédiction de maladie cardiaque basé sur le **Heart Disease Dataset** (Kaggle).  
Il comprend la préparation des données, l’analyse exploratoire (EDA), la modélisation avec Random Forest et l’évaluation des performances.

---

##  Contenu du dépôt
- `rapport.md` – Rapport complet  
- `code.ipynb` – Code Python  
- Figures produites : ROC, matrice de confusion, matrice de corrélation, distribution de l’âge  

---

##  Dataset
Dataset de **303 patients** avec **14 variables** : âge, sexe, tension artérielle, cholestérol, fréquence cardiaque maximale, ECG, angine d’effort, oldpeak, slope, ca, thal, et `target` (0 = sain, 1 = malade).

---

##  Modèle
Modèle utilisé : **Random Forest Classifier**  
- `n_estimators = 100`  
- `random_state = 42`  
Très robuste et efficace pour ce type de données médicales.

---

##  Résultats
- **Accuracy : > 95 %**  
- **AUC ROC : 0.995**  
- Très faible taux d’erreurs  
- Excellente capacité de discrimination entre malades et non malades

---

##  Conclusion
Le modèle présente d’excellentes performances et montre le potentiel du machine learning pour la prédiction du risque cardiaque. Une validation sur des données externes est recommandée pour confirmer la généralisabilité.

---

## Lien video

https://drive.google.com/file/d/1gzOKtOwHrWUgFX0-lyhaf78LVNYEhbaL/view?usp=sharing
