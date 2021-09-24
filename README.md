# 2021_MSA_Submission

### _Dependencies alongside their versions_
* python = ">=3.7.1, <3.9"
* joblib = "^1.0.1"
* json5 = "^0.9.6"
* sklearn = "^0.0"
* missingno = "^0.5.0"
* nltk = "^3.6.3"
* azureml-defaults = "^1.34.0"
* pandas = "^1.3.3"
* bs4 = "^0.0.1"
* requests = "^2.26.0"

### _Files Included_
* CommonLit_Readability_Prize.ipynb - jupyter notebook containing all the steps for steps 1 to 4 outlined in the Phase 2 brief
* MSA_Report.pdf - Report for step 4 outlined in the Phase 2 brief
* echo_score.py - scorer file for deployment of ML model to Azure
* config.json - file o configure Azure environemt
* model.pkl - pickle file containing the details for the ML model used
* kaggle_submission.csv - csv file containing the data provided for the commonLit competition
* train.csv - preprocessed training data used by echo_score.py to fit the TFIDF transformer in the Azure environement
* books.csv - data of top 100 books gathered from gutenberg's website
* labels.csv - data containing the labels of the ML model
* LUIS.json - JSON file containing the details for provisioning the chatbot

### _Folder included_
* kaggle_competition - folder containing .csv files provided for the commonLit competition

### _Azure Bot channel Username_
* Telgram - @MSA2021ReadingEaseBot
