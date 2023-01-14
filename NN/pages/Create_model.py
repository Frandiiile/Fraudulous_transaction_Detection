import numpy as np
import pickle
import streamlit as st
import pandas as pd
import pickle
st.title("Create your own model")
st.subheader("Algorithm")
chart_select = st.selectbox(
        label='Choose one',
        options=['SVM','XGBoost']
    )
data1=pd.read_csv("data\ML.csv")
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
X = data1[['amountofmoney','typeofaction','hour','minute']]
y= data1['isfraud']
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_res, y_res = sm.fit_resample(X, y.ravel())

if chart_select == 'SVM':
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
    model=SVC()
    model.fit(X_train, y_train)
    def svm_model(gamma, C):
        svm = SVC(kernel='rbf',gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        pickle_out = open("classifier6.pkl","wb")
        pickle.dump(svm, pickle_out)
        pickle_out.close()
        model=svm
        return accuracy_score(y_test, y_pred)
    gamma = st.slider('Choose your gamma', 0.00001, 1.00, 0.2)
    C = st.slider('Choose your C', 0.1, 1000.00, 1.00)
    if st.button('Model Creation Result'):
        acc = svm_model(gamma, C)
        st.write('Your model\'s accuracy is: ', acc)
if chart_select =='XGBoost':
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
    model=XGBClassifier()
    model.fit(X_train, y_train)
    def xgb_model(learning_rate, max_depth, n_estimators):
        xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        pickle_out = open("classifier6.pkl","wb")
        pickle.dump(xgb, pickle_out)
        pickle_out.close()
        model=xgb
        return accuracy_score(y_test, y_pred)
    learning_rate= st.number_input('Choose your learning rate')
    max_depth= st.slider('Choose you max depth', 3, 10, 1)
    n_estimators = st.slider('Number of estimators (n_estimators)',100,1000,100)
    if st.button('Model Creation Result'):
        acc = xgb_model(learning_rate, max_depth, n_estimators)
        st.write('Your model\'s accuracy is: ', acc)

import pickle
pickle_out = open("classifier6.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# loading the saved model
loaded_model = pickle.load(open('classifier6.pkl','rb'))


# creating a function for Prediction

def transaction_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The transaction is not fraudulous'
    else:
      return 'The transaction is fraudulous'
  
    
  
def main():
    
    
    # giving a title
    st.title('Fraudulous Transactions Prediction Web App')
    st.header("Predict with your Model")
    
    
    # getting the input data from the user
    
    
    TypeofAction= st.radio('Choose the type of action (0 is Transfer and 1 is Cash out)',(0,1))
    Amount= st.number_input('The amount of money to be transferred')
    Hour = st.number_input('Hour of the transaction')
    Minute = st.number_input('Minute of the transaction')
   
    
    # code for Prediction
    pred = ''
    
    # creating a button for Prediction
    
    if st.button('Model Prediction Result'):
        pred = transaction_prediction([TypeofAction, Amount, Hour, Minute])
        
        
    st.success(pred)
if __name__ == '__main__':
    main()