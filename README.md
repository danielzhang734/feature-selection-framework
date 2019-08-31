# feature-selection-framework
This project includes data type detection and feature selection.  
The part of data type detection can detect data type into 5 different types such as Numerical, Nominal, Datetime, Multi-plain, and Language text.  
After finishing data type detection. The saved Json file will be used for Feature Selection.  
Test.py: The drive file to run the project of detecting data type.   
finalized_model_test_BT_V2.sav: Saved trained Model for Data Type Detection 
Feature selection:  The file to rank and ignore features 
FeatureSelection = FeatureSelection(dataset,json)   #eg : dataset ='dataset.csv' json = 'datatypes.json'  
For the FeatureSelection object, it has six methods which you can use it separately or together:  
FeatureSelection.select_missing()  
FeatureSelection.select_select_lowvariance()  
FeatureSelection.select_pearson()  
FeatureSelection.select_chi2()  
FeatureSelection.select_L1()   
FeatureSelection.select_tree()  
  
FeatureSelection.feature_selection() #together and return a list and dict which contain the names of the features being ignored  

And you can use following variable:  
  
FeatureSelection.X_NOMINAL   
FeatureSelection.X_REAL   
FeatureSelection.X_DATE   
FeatureSelection.X_MULTI   
FeatureSelection.X_NATURAL   
FeatureSelection.Y # the label  
FeatureSelection.X_IG  # the original dataset without label.(in this case, the date has been transformed into three columns including : 'day_of_month', 'day_of_week' and 'week_of_year')  
FeatureSelection.ignore #the dict which contains the features being ignored  
