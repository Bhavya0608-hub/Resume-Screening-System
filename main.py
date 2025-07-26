import streamlit as st
import pickle
import re
import nltk


nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf= pickle.load(open('clf.pkl', 'rb'))
tfidf= pickle.load(open('tfidf.pkl', 'rb'))

def CleanResume(txt):
  CleanTxt= re.sub('http\S+\s','',txt)
  CleanTxt= re.sub('RT|cc',' ',CleanTxt)
  CleanTxt= re.sub('@\S+',' ',CleanTxt)
  CleanTxt= re.sub('#\S+',' ',CleanTxt)
  CleanTxt= re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',CleanTxt)
  CleanTxt= re.sub(r'[^\x00-\x7f]',' ',CleanTxt)
  CleanTxt= re.sub('\s+',' ',CleanTxt)
  return CleanTxt



#web
def main():
    st.title("Resume Screening System")
    upload_file= st.file_uploader('Upload Resume', type=['txt','pdf'])

    if upload_file is not None:
        try:
            resume_bytes= upload_file.read()
            resume_text= resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            resume_text= resume_bytes.decode('latin-1')

        cleaned_resume= CleanResume(resume_text)
        cleaned_resume= tfidf.transform([cleaned_resume])
        prediction_id= clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

        category_mapping = {
            15: 'Java Developer',
            23: 'Testing',
            8: 'DevOps Engineer',
            20: 'Python Developer',
            24: 'Web Designing',
            12: 'HR',
            13: 'Hadoop',
            3: 'Blockchain',
            10: 'ETL Developer',
            18: 'Operations Manager',
            6: 'Data Science',
            22: 'Sales',
            16: 'Mechanical Engineer',
            1: 'Arts',
            7: 'Database',
            11: 'Electrical Engineering',
            14: 'Health and fitness',
            19: 'PMO',
            2: 'Business Analyst',
            9: 'DotNet Developer',
            2: 'Business Analyst',
            17: 'Network Security Engineer',
            21: 'SAP Developer',
            5: 'Civil Engineer',
            0: 'Advocate',
            4: 'Automation Testing',
        }

        category_name = category_mapping.get(prediction_id, 'Unknown')
        st.write("Predicted Category: ", category_name)


# python main
if __name__== "__main__":
    main()

