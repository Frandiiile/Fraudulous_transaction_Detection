import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('classifier5.pkl','rb'))


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
    st.header("KNN Algorithm")
    st.subheader("What is it?")
    st.write("KNN is a supervised machine learning algorithm used for classification and regression that classifies data points by finding the nearest neighbors.")
    
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