# AIQuest_healthcare_labs
In this problem Statement by reading it. input_specialties.csv file has 10050 rows where 3733  distinct coming .Like Internal_Medicine coming 112 times ,Family Medicine coming 93 times,Cardiology coming 60 times,and Neurology coming 56 times etc. 
"""""""""""""""
df = pd.read_csv("input_specialties.csv")
value_counts = df['raw_specialty'].value_counts()

df = pd.DataFrame({'Answer': value_counts})
""""""""""""""""""
