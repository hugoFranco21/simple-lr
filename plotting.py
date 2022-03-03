import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from jsonschema import validate
import numpy

coefficients = [15474.641917266252, 40434.828554699525, 41070.89890345359, 40333.321557077004, 18848.58522068252, 16208.389893117124, 22243.339231275633, 24105.90611545494, 4164.772926590216, 1545.66504043396, 7713.376000750904, 4867.980804174962, 3660.479537770913, 39414.38519263655, -11723.44287074937, 7510.605980978513, 19310.262715758203, 4948.1351321853645]
min_max_scaler = MinMaxScaler()
min_max_scaler.scale_ = [0.25, 0.25, 1,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
    1.,   1.,   1., 1.,   1. ,   1. ,  1.  ]
min_max_scaler.min_ = [-0.25, -0.25,  0.,    0.,    0.,    0.,    0.,    0.,    0.,
    0.,    0.,    0.,  0.,    0.,    0,    0. ,   0.,    0.  ]
min_max_scaler.clip = False

columns = ['Job Title', 'Gender', 'Age', 'Performance Evaluation', 'Education', 'Department', 'Seniority', 'Base Pay', 'Bonus']

def prepare_data():
	df = pd.read_csv('datasets/glassdoor.csv', header=1, names = columns)
	return df

def h(params, sample):
	"""This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 
	Returns:
		Evaluation of h(x)
	"""
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum

def main():
    df = prepare_data()
    df['TC'] = df['Bonus'] + df['Base Pay']
    df_y = df['TC']
    df_x = df[['Job Title', 'Gender', 'Performance Evaluation', 
    'Education', 'Department', 'Seniority']]
    one_hot_prof = pd.get_dummies(df['Job Title'])
    one_hot_dep = pd.get_dummies(df['Department'])
    one_hot_gender = pd.get_dummies(df['Gender'])
    one_hot_educ = pd.get_dummies(df['Education'])
    #print(one_hot_dep)
    #print(one_hot_prof)
    df_x = df_x.drop('Job Title', axis = 1)
    df_x = df_x.drop('Department', axis = 1)
    df_x = df_x.drop('Gender', axis = 1)
    df_x = df_x.drop('Education', axis = 1)
    #print(df_x.head())
    df_x = pd.concat([df_x, one_hot_gender, one_hot_educ, one_hot_prof], axis = 1)
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sort_values('TC')
    df_x = df.drop('TC', axis=1, inplace=False)
    df_y = df['TC']
    Y_pred = []
    X = df_x.to_numpy()
    norm_xtest = min_max_scaler.transform(X)
    for x in norm_xtest:
        aux = h(coefficients, x)
        Y_pred.append(aux)
    print(df.head())
    plt.plot(range(1, len(df_y) + 1), df_y.to_numpy())
    plt.plot(range(1, len(df_y) + 1), Y_pred)
    plt.savefig('assets/comparison.png')

main()
