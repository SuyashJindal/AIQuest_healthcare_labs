# AIQuest_healthcare_labs
In this problem Statement by reading it. input_specialties.csv file has 10050 rows where 3733  distinct coming .Like Internal_Medicine coming 112 times ,Family Medicine coming 93 times,Cardiology coming 60 times,and Neurology coming 56 times etc. 

"""""""""""""""
  df = pd.read_csv("input_specialties.csv")
  value_counts = df['raw_specialty'].value_counts()

  df = pd.DataFrame({'Answer': value_counts})
""""""""""""""""""
#Not used Pretrained Hugging face model (like BioBert) or heavy transformer model  due to time complexity in running time streaming data which takes time and Space complexity like billion of parameters (MBs in size of file)
 nucc_taxonomy_master.csv file has 879 distinct values. Grouping column has 29 distinct values. Classification column has 245 distinct names. Specialization has 477 (with 240 missing) !Display_Name  has all 879 distinct names.
 In this I have tries several approached and Models (statistically ) focused and NLP based.
 
