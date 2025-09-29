import pickle
import pandas as pd
import streamlit as st

with open("temp_book.pkl","rb") as f:
    tempe=pickle.load(f)
with open("cosin_book.pkl","rb") as f1:
    cosineva1=pickle.load(f1)
def get_book_semlarty(book_namep,cosineva=cosineva1):
    matches = tempe[tempe['Book_Name'].str.strip().str.lower() == book_namep.strip().lower()]
    if matches.empty:
        return f"No author found matching '{book_namep}'"
    id=matches.index[0]
    simlarty=list(enumerate(cosineva[id]))
    simlarty=sorted(simlarty,key=lambda x: x[1],reverse=True)
    simlarty=simlarty[1:5]
    book_index=[i[0] for i in simlarty]
    simlarty=tempe[['Book_Name', 'Book_Author', 'Book_Image']].iloc[book_index]
    return simlarty
st.title("Book Recomendetion System;")
book_name=st.selectbox("enter your book name",tempe['Book_Name'].values)
if st.button("Show recomndation"):
    sample=get_book_semlarty(book_name)
    st.write("top 10 recomendation movies")
    for i in range(0, 6, 2):
        col=st.columns(2)
        for col ,j in zip(col,range(i,i+2)):
            if j<len(sample):
                Book_author=sample.iloc[j]['Book_Author']
                poster_path=sample.iloc[j]['Book_Image']
                Book_name=sample.iloc[j]['Book_Name']
                with col:
                    st.write(Book_author)
                    st.image(poster_path,width=150)
                    st.write(Book_name)
                


