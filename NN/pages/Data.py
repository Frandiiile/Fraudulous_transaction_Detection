import streamlit as st
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def main():
    st.title("      ML Project")
    st.write("""
    # Money Laundering Detection

    """)
    st.write('---')
    st.subheader('Dataset')
    data1=pd.read_csv("data\ML.csv")
    st.write(data1)
    st.subheader("Data Preparation")
    data1['date']=pd.to_datetime(data1['date'])
    data1['year']=data1['date'].dt.year
    data1['month']=data1['date'].dt.month
    data1['day']=data1['date'].dt.day
    data1['hour']=data1['date'].dt.hour
    data1['minute']=data1['date'].dt.minute
    data1['typeofaction']=data1['typeofaction'].astype('category')
    data1['typeofaction']=data1['typeofaction'].cat.codes
    data1['typeoffraud']=data1['typeoffraud'].astype('category')
    data1['typeoffraud']=data1['typeoffraud'].cat.codes
    data1.drop('typeoffraud',axis=1,inplace=True)
    st.write(data1)
    st.subheader("Type of chart")
    chart_select = st.selectbox(
        label='Choose one',
        options=['Scatterplots','Correlation']
    )

    numeric_columns = list(data1.select_dtypes(['float','int']).columns)

    if chart_select == 'Scatterplots':
        st.subheader('Scatterplot Settings')
        try:
            x_values = st.selectbox('X axis',options=numeric_columns)
            y_values = st.selectbox('Y axis',options=numeric_columns)
            plot = px.scatter(data_frame=data1,x=x_values,y=y_values)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select =='Correlation':
        st.subheader('Matrice de Correlation')
        try:
            fig, ax = plt.subplots()
            sns.heatmap(data1.corr(), ax=ax)
            st.write(fig)
        except Exception as e:
            print(e)
    st.subheader('Feature Selection')
    st.write('Choosing the data to work with is a key factor determining the quality of your work')
    st.write('Let\'s see how to choose the best features')
    st.write('There a re manual selections (from our observations) and selections based on feature importances scores we\'ll leave you with the feature importances scores')
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = data1[['amountofmoney','typeofaction','year','month','day','hour','minute']]
    y= data1['isfraud']
    bestfeatures = SelectKBest(score_func=chi2, k=7)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    st.write(featureScores.nlargest(10,'Score'))  #print 10 best features

if __name__ == '__main__':
    main()
