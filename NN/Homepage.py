import streamlit as st

# TypeError: can't pickle module objects
# Custom imports 
from multipage import MultiPage
from pages import Create_model,Data, XGBoost, KNN, SVM,RandomForest# import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("Fraudulous Transactions Prediction Web App")
st.subheader("Why target Fradulous transactions?")
st.write(" \tFraudulent actions are any form of intentional deception or misrepresentation of facts with the purpose of achieving some kind of financial or other gain. Examples of fraudulent actions include identity theft, financial fraud, cybercrime, and more.\n"
"\n \tFraudulent actions are any form of intentional deception or misrepresentation of facts with the purpose of achieving some kind of financial or other gain. Examples of fraudulent actions include identity theft, financial fraud, cybercrime, and more.\n"
"\n \tFor these reasons, it is important to detect and prevent fraudulent actions. Businesses and organizations can help protect themselves and their customers by implementing measures such as identity verification, background checks, and data security protocols. Additionally, individuals should be aware of potential signs of fraudulent activity and report any suspicious activity to the authorities. By detecting and stopping fraudulent actions, everyone can benefit from a safer, more secure environment.")

st.markdown("Choose the model you want to use or visualize the data")

# Add all your applications (pages) here
app.add_page("Dataset work", Data.main)
app.add_page("Random Forest", RandomForest.main)
app.add_page("XGBoost", XGBoost.main)
app.add_page("SVM",SVM.main)
app.add_page("KNN",KNN.main)
app.add_page("Create your own Model", Create_model.main)
app.run()