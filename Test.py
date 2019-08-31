'''
# Drive to run the Data Type Detection
dataset: Dataset need to be tested for data type
model: trained model (no need to change)
label: use for training and test (set as None)
size: the percentage of the dataset to be detected. the number in (0,1]. 
      Default be 1. change it if the dataset is too large (long time to run)
The detected data type will be saved in Json format file
'''
from summaryTable_test import SummaryTest
import time

start_time = time.time()
testing = SummaryTest()
dataset = 'E:\DATAmode\smartdata.csv'
model = 'finalized_model_test_BT_V2.sav'
label = None
size = 0.2
testing.summary_table(dataset,model,label,size)
print("--- %s seconds ---" % (time.time() - start_time))