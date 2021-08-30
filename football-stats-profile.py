import pandas
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

"""'
Independent Variables (Features) = Variables that we are going to use to predict (ex.: possession, shots, etc)
Dependent Variable = Variables that we want to predict (final score)
"""

STATS_FILE = "resources/Chelsea-Stats.csv"
FEATURES = [
    "Poss",
    "Touches",
    "Touches Def Pen",
    "Touches Def 3rd",
    "Touches Mid 3rd",
    "Touches Att 3rd",
    "Touches Att Pen",
    "Live Touches",
    "Dribbles Succ",
    "Dribbles Att",
    "Dribbles Succ%",
    "Dribbles #Pl",
    "Dribbles Megs",
    "Carries",
    "Carries TotDist",
    "Carries PrgDist",
    "Carries Prog",
    "Carries 1/3",
    "Carries CPA",
    "Carries Mis",
    "Carries Dis",
    "Receiving Targ",
    "Receiving Rec",
    "Receiving Rec%",
    "Receiving Prog"
]
DEPEND_VARIABLE = "Result"

# Reading Dataset
raw_df = pandas.read_csv(STATS_FILE, sep=";")

# Structuring the dataset
raw_df[DEPEND_VARIABLE] = raw_df["GF"] - raw_df["GA"]
# raw_df.loc[raw_df[DEPEND_VARIABLE] == "W", DEPEND_VARIABLE] = 1
# raw_df.loc[raw_df[DEPEND_VARIABLE] == "D", DEPEND_VARIABLE] = 0
# raw_df.loc[raw_df[DEPEND_VARIABLE] == "L", DEPEND_VARIABLE] = -1
df = raw_df.filter(FEATURES + [DEPEND_VARIABLE], axis=1)
print(df.head())

# # Defining variables
independent_variables = df.drop([DEPEND_VARIABLE], axis=1).values
dependent_variable = df[DEPEND_VARIABLE].values
# print(independent_variables)
# print(dependent_variable)

# Creating test and training dataset
(
    independent_variables_train,
    independent_variables_test,
    dependent_variable_train,
    dependent_variable_test,
) = train_test_split(independent_variables, dependent_variable, test_size=0.3, random_state=0)

# # Training
regression = SGDRegressor()
regression.fit(independent_variables_train, dependent_variable_train)

# # Evaluating
prediction = regression.predict(independent_variables_test)
score = r2_score(dependent_variable_test, prediction)
print("Model score: ", score)
print("Coef: ", regression.coef_)
# coefficients = {}
# for coef, feature in zip(regression.coef_, FEATURES):
#     coefficients[feature] = coef
#     print(f"{feature} -> {coef}")
