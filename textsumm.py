import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def _input():
    article = st.text_input("Enter text")
    #"It is an encouraging time to be an international student in the U.S. Two recent developments – the election of Joe Biden as U.S. President and the advent of highly effective COVID-19 vaccines – will result in more welcoming policies to international students and return the U.S. to its historical place as the #1 most popular destination for high-quality students from around the globe. Based on the trends we see here at U.S. News Global Education, we recommend that international students consider:  1) Enrolling in spring, rather than waiting for fall, if you want a better chance of getting into the university of your choice. Even the most prestigious U.S. universities have seen large percentages of deferrals this past year due to the coronavirus. There are always more qualified applicants at top universities than can be accepted, but this spring represents a unique opportunity for undergraduate and graduate students alike. If you are willing to do one semester online this spring (and many universities have rolled out exceptional virtual classroom offerings), you are likely to be accepted at universities that may not have space if you wait until Fall 2021. 2) Planning for an on-campus fall semester in the U.S. American universities have been highly effective at containing the virus on their campuses this past semester and most expect to welcome a larger number of in-person students this spring. With vaccine distribution beginning this month and with President-elect Biden expected to assure visa stability for international students soon after taking office on January 20, 2021, being able to come to campus this fall seems nearly assured. We expect President-elect Biden to begin issuing executive orders soon after taking office that will re-open travel and re-staff U.S. counselor offices so that plans can be made and visas can be issued well in advance of the Fall 2021 semester. 3) Preparing to see better job prospects for international students in the U.S. Supporters of Optical Practical Training (OPT) in the United States received a major victory when a federal judge supported the U.S. Department of Homeland Security, the U.S. Chamber of Commerce, and several other employer- and university groups recently by striking down a years-long lawsuit from a group arguing that the OPT program should be eliminated. Just days later, the U.S. Senate voted 100-0 to unanimously pass the Fairness for Highly Skilled Immigrants Act. The legislation will significantly benefit thousands of international professionals working in the United States after receiving their degrees. After a series of calls with global leaders last month, President-Elect Joe Biden declared: “I am letting them know that America is back.” When it comes to international student growth and success, U.S. News Global Education agrees with the President-Elect. The United States is about to re-emerge as the #1 destination for international higher education. For students, parents and counselors across the world, this is exciting and welcome news."
    return article

def main():
    article = _input()
    length = st.number_input("Enter the number of required sentences",min_value=1,format="%d")
    if st.button("Submit"):
            required_length = length
            tokenized_article = sent_tokenize(article)
            cleaned_article = clean(tokenized_article) 
            probability_dict = init_probability(cleaned_article)
            sentence_weights = average_sentence_weights(cleaned_article,probability_dict)
            summary = generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,required_length)
            st.write(summary)
    else:
        st.write("Hi!")

def clean(sentences):
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
        sentence = sentence.split()
        sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in set(stopwords.words('english'))]
        sentence = ' '.join(sentence)
        cleaned_sentences.append(sentence)
    return cleaned_sentences

def init_probability(sentences):
    probability_dict = {}
    words = word_tokenize('. '.join(sentences))
    total_words = len(set(words))
    for word in words:
        if word!='.':
            if not probability_dict.get(word):
                probability_dict[word] = 1
            else:
                probability_dict[word] += 1

    for word,count in probability_dict.items():
        probability_dict[word] = count/total_words 
    
    return probability_dict

def update_probability(probability_dict,word):
    if probability_dict.get(word):
        probability_dict[word] = probability_dict[word]**2
    return probability_dict

def average_sentence_weights(sentences,probability_dict):
    sentence_weights = {}
    for index,sentence in enumerate(sentences):
        if len(sentence) != 0:
            average_proba = sum([probability_dict[word] for word in sentence if word in probability_dict.keys()])
            average_proba /= len(sentence)
            sentence_weights[index] = average_proba 
    return sentence_weights

def generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,summary_length = 30):
    summary = ""
    current_length = 0
    while current_length < summary_length :
        highest_probability_word = max(probability_dict,key=probability_dict.get)
        sentences_with_max_word= [index for index,sentence in enumerate(cleaned_article) if highest_probability_word in set(word_tokenize(sentence))]
        sentence_list = sorted([[index,sentence_weights[index]] for index in sentences_with_max_word],key=lambda x:x[1],reverse=True)
        summary += tokenized_article[sentence_list[0][0]] + "\n"
        for word in word_tokenize(cleaned_article[sentence_list[0][0]]):
            probability_dict = update_probability(probability_dict,word)
        current_length+=1
    return summary


if __name__ == "__main__":
    st.title("Text Summarization")
    main()
