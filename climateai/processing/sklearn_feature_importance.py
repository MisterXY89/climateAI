
import matplotlib.pyplot as plt

def plot_sk_feature_importance(sk_model, X):
    # Get the feature importances
    try:
        ft_importances = sk_model.feature_importances_
    except Exception as e:
        ft_importances = sk_model.coef_

    # Print the feature importances
    print(ft_importances)

    # Sort the features by their importance
    sorted_idx = ft_importances.argsort()[::-1]

    # Get the names of the features
    feature_names = X.columns[sorted_idx]

    # Create a bar plot of the feature importances
    plt.bar(range(X.shape[1]), ft_importances[sorted_idx], 
        color = "#0097a7"
    )
    plt.xticks(range(X.shape[1]), feature_names, rotation=90)
    plt.show()
