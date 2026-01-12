from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# --------------------------------------------------
# Yapay Zeka Modelleri
# --------------------------------------------------
def get_models():
    return {
        "Doğrusal Regresyon": LinearRegression(),
        "Bayesçi Doğrusal Regresyon": BayesianRidge(),
        "Karar Ağacı Regresyonu": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting Regresyonu": GradientBoostingRegressor(random_state=42),
        "Yapay Sinir Ağı": MLPRegressor(
            hidden_layer_sizes=(50, 50),
            max_iter=1000,
            random_state=42
        )
    }
