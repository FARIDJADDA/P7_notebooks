# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    """
    Applique l'encodage one-hot aux variables catégorielles du DataFrame.

    Args:
    df (pd.DataFrame): DataFrame à encoder.
    nan_as_category (bool): Indique si les NaN doivent être traités comme une catégorie.

    Returns:
    pd.DataFrame: DataFrame avec variables catégorielles encodées.
    list: Liste des nouvelles colonnes générées.
    """
    # Sauvegarde des noms de colonnes originaux
    original_columns = list(df.columns)
    # Identification des colonnes catégorielles
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    # Application de l'encodage one-hot
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    # Extraction des nouvelles colonnes ajoutées par l'encodage
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


# Flat index renaming
def flatten_and_rename_columns(grouped_df, prefix=''):
    """
    Aplatit un index hiérarchique de DataFrame et renomme les colonnes avec un préfixe et la méthode d'agrégation.
    
    Args:
    grouped_df (pd.DataFrame): DataFrame avec un MultiIndex en colonnes, typiquement issu d'un groupby().agg().
    prefix (str): Préfixe à ajouter devant chaque nom de colonne pour personnaliser et clarifier les noms.

    Returns:
    pd.Index: Un nouvel index pour les colonnes du DataFrame, avec les noms de colonnes modifiés.
    """
    return pd.Index([prefix + elem[0] + "_" + elem[1].upper() for elem in grouped_df.columns.tolist()])
