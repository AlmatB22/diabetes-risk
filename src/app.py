import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Input, Output
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import catboost
from catboost import Pool, CatBoostClassifier
import plotly.express as px
import pickle
import joblib


app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
server = app.server

url_dataset = 'https://raw.githubusercontent.com/AlmatB22/diabetes_data/master/diabetes_data.csv'

df_all = pd.read_csv(url_dataset,sep=",")

df_all.head()

df_all['gender'].replace(['Male', 'Female'], [1,0], inplace=True)

Y_col = 'class' # this is output column

X_cols = df_all.loc[:, df_all.columns != Y_col].columns # input, anything except output (Y_col)

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df_all[X_cols], df_all[Y_col],test_size=0.2, random_state=42, shuffle=True)
cat_columns = ['gender'] 
categorical_features = [1]


model = pickle.load(open('model_pkl', 'rb'))


explainer = joblib.load('explainer.bz2')

shap_values = explainer.shap_values(Pool(X_test, y_test, cat_features=categorical_features))
d2 = pd.DataFrame(columns = ['age',	'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity'])
dic = {"age": 0,	"gender": 0, "polyuria":0, "polydipsia":0, "sudden_weight_loss": 0, "weakness": 0, "polyphagia":0, "genital_thrush":0, "visual_blurring":0, "itching":0, "irritability":0, "delayed_healing":0, "partial_paresis":0, "muscle_stiffness":0, "alopecia":0, "obesity":0}
resultat = {"age": 0,	"gender": 0, "polyuria":0, "polydipsia":0, "sudden_weight_loss": 0, "weakness": 0, "polyphagia":0, "genital_thrush":0, "visual_blurring":0, "itching":0, "irritability":0, "delayed_healing":0, "partial_paresis":0, "muscle_stiffness":0, "alopecia":0, "obesity":0}


for patient in range(len(X_test)):
    prediction = model.predict_proba(X_test.iloc[[patient], :])
    label = np.argmax(prediction, axis = 0)
    print('Label : ', label , "prediction: ", prediction)
    print(patient)
    sdf_train = pd.DataFrame(
        {
            'feature_value': X_test.iloc[[patient], :].values.flatten(),
            'shap_values': shap_values[[patient]].flatten()
        }
    )
  
sdf_train.index = X_test.columns.to_list()
sdf_train['Feature'] = sdf_train.index

aux = dict(zip(sdf_train.Feature, sdf_train.shap_values))
dic.update(resultat)
resultat = {**dic, **aux}
  
for key,value in aux.items():
    resultat[key] = np.abs(value) + dic[key]

resultat = dict(reversed(sorted(resultat.items(), key=lambda item: item[1])))

keys = list(resultat.keys())
values = list(resultat.values())

fd = pd.DataFrame.from_dict(resultat, orient ='index')
fd['Feature'] = fd.index
fd.rename(columns={0: 'shap_values'}, inplace=True, errors='raise')
 



app.layout = dbc.Container([


    dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        (dbc.NavbarBrand("Recommendation Framework for Online Stratification of Diabetes Risk", className="text-center")),
                    ],
                    align="center",
                    
                ),
                href="/",
            ),
        ],
        
    ),
  
    color="primary",
    dark=True,
    className="mb-5 text-center",
    ),
 

    

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Label('', className = 'mb-4'),
                html.Label('Dropdown'),
                html.H6('Global - explanation for the whole dataset', className="blockquote-footer"),
                html.H6('Local - explanation for only one data sample', className="blockquote-footer"),
                dcc.Dropdown(options= ['Global', 'Local'],value = 'Global', id = 'radio_item', className='mb-4'),
                
                html.Label('Patient ID'),
                dcc.Input(
                    value = 0, 
                    type= 'number', 
                    id = 'patient_id',
                    className='mb-4',
                

                )   
             ], ),
            
            dbc.Row([
                html.Label('Result', className='mb-3'),
                html.H3("", id = 'class_output', className= 'text-md-center')
            ])
            
            
        ],width = {'size': 3, 'offset': 1}, className = 'shadow shadow-white'),
        dbc.Col([
            dcc.Graph(
                figure = {},
                id = 'graph'
            )
        ], width={'size': 6, 'offset':1}, className = 'shadow shadow-white'),
        
    ])
], 
fluid=True)   

@callback(
    
    Output(component_id= 'graph', component_property='figure'),
    Output(component_id='class_output', component_property='children'),
    [Input(component_id= 'radio_item', component_property='value'),
     Input(component_id = 'patient_id', component_property = 'value')]
    
)
def update_graph(mode,id):
    if mode == 'Global':
        figure = px.histogram(fd, x= 'Feature', y = 'shap_values', title= 'SHAP Explanation ')
        output_text =  'pick local to find out the diabetes risk'
        
    else:
        sdf_train = sdf_train = pd.DataFrame({
            'feature_value': X_test.iloc[[id],:].values.flatten(),
            'shap_values': shap_values[id].flatten()
        })
        sdf_train.index =X_test.columns.to_list()
        sdf_train = sdf_train.sort_values('shap_values',ascending=False)  
        sdf_train['Feature'] = sdf_train.index
        prediction = model.predict(X_test.iloc[[id], :])
        figure =px.histogram(sdf_train,x = 'Feature', y = 'shap_values', title= "SHAP Explanation ")
        if prediction[0] == 0:
            output_text = 'There is no diabetes risk'
        else:
            output_text = 'There is a diabetes risk'
        
    
    return figure, output_text
        

if __name__ == '__main__':
    app.run_server(debug=True)