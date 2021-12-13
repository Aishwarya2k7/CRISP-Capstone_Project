from flask import Flask,render_template, request, redirect, url_for, flash
from wtforms.validators import Length, DataRequired
from youtube_transcript_api import YouTubeTranscriptApi
import goslate
from french_lang import frontend_freeng

import numpy as np
import pandas as pd
import nltk
import nltk.data
#<nltk.download('punkt')> # one time execution
#nltk.download('stopwords')
import re
#from nltk.corpus import stopwords
from flask_mobility import Mobility
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
translation1 = pipeline("translation_en_to_de")
#translation2 = pipeline("translation_en_to_fr")
translation3 = pipeline("translation_en_to_ro")

import moviepy.editor as mp
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks

app = Flask(__name__)
app.config['SECRET_KEY'] = '25999940e4e2ea95bce98d7b'
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, IntegerField, SelectField, SubmitField

class HomeForm(FlaskForm):
    option1 = SubmitField(label='Enter Videoid')
    option2 = SubmitField(label='Upload Video')

class EntryForm(FlaskForm):
    videoid = StringField(label='Video ID', validators=[Length(min=11, max=11), DataRequired()])
    numlines = IntegerField(label='Number of Lines', validators=[DataRequired()])
    language = SelectField(label='Choose language', choices = [('German', 'German'), ('French', 'French'), ('Dutch','Dutch'),('Romanian', 'Romanian'), ('Hindi', 'Hindi')], validators=[ DataRequired()])
    submit = SubmitField(label='Submit')


class Entry2Form(FlaskForm):
    #videoid = StringField(label='Video Id', validators=[Length(min=11, max=11), DataRequired()])
    video = FileField(label='Upload', validators=[DataRequired()])
    numlines = IntegerField(label='Number of Lines', validators=[DataRequired()])
    language = SelectField(label='Choose language', choices = [('German', 'German'), ('French', 'French'),('Dutch','Dutch'), ('Romanian', 'Romanian'), ('Hindi', 'Hindi')], validators=[ DataRequired()])
    submit = SubmitField(label='Submit')

@app.route('/')

@app.route('/home',methods=['GET','POST'])
def home_page():
    return render_template("home.html")


@app.route('/help')
def help_page():
    return render_template("help.html")


@app.route('/output')
def output():
    return render_template("output.html")

@app.route('/entry2',methods=['GET','POST'])
def entry2_page():
    form = Entry2Form()

    '''def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        video = request.files['video']
        # if user does not select file, browser also
        # submit an empty part without filename
        if video.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if video and allowed_file(video.filename):
            filename = video.filename
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',
                                    filename=filename))'''
    if form.validate_on_submit():
        video = form.video.data
        numlines = form.numlines.data
        language = form.language.data
        #vid.save(os.path.join(app.config['vid'], video))
        #print(video)'''

        try:

            my_path="/Users/aishw/Desktop/"
            my_clip = mp.VideoFileClip(r""+my_path+video)

            my_clip.audio.write_audiofile(r""+my_path+"my_res.wav")

            r = sr.Recognizer()

            # a function that splits the audio file into chunks
            # and applies speech recognition
            def get_large_audio_transcription(path):

                '''Splitting the large audio file into chunks
                and apply speech recognition on each of these chunks'''

                myaudio = AudioSegment.from_file(path)
                chunk_length_ms = 10000 # pydub calculates in millisec
                chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of ten secs


                folder_name = "audio-chunks"
                # create a directory to store the audio chunks
                if not os.path.isdir(folder_name):
                    os.mkdir(folder_name)
                whole_text = ""
                # process each chunk
                for i, audio_chunk in enumerate(chunks, start=1):
                    # export audio chunk and save it in
                    # the `folder_name` directory.
                    chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
                    audio_chunk.export(chunk_filename, format="wav")
                    # recognize the chunk
                    with sr.AudioFile(chunk_filename) as source:
                        audio_listened = r.record(source)
                        # try converting it to text
                        try:
                            text = r.recognize_google(audio_listened)
                        except sr.UnknownValueError as e:
                            #print("Error:", str(e))
                            whole_text="Sorry, audio not clear."
                        else:
                            text = f"{text.capitalize()}."
                            #print(chunk_filename, ":", text)
                            whole_text += text
                # return the text for all chunks detected
                #print(whole_text)
                return whole_text


            path = my_path+"my_res.wav"

            text1 = get_large_audio_transcription(path)
            #print(transcript)

        except:
            text1='No trancript available.'
            numlines=0


        if(int(numlines)>len(text1)):
            numlines=len(text1)

        #mytext = [" ".join(t.split("\n")) for t in text]
        #text = " ".join(mytext)
        text=text1.split('.')
        #text= laugh_removal(text1)
        #print(text)

        word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs

        f.close()
        # remove punctuations, numbers and special characters
        #clean_sentences = pd.Series(text).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in text]


        #stop_words = stopwords.words('english')
        #clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
            sim_mat = np.zeros([len(text), len(text)])
        for i in range(len(text)):
            for j in range(len(text)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(text)), reverse=True)

        #ranked_sentences = list(set(ranked_sentences)

        '''ttext1=ranked_sentences[:numlines//2]
        ttext2=ranked_sentences[((numlines//2)+1):numlines]

        ttext11= str(ttext1).strip('[]')
        ttext12= ''.join(filter(lambda x: x.isalpha() or x.isspace(), ttext11))

        ttext21= str(ttext2).strip('[]')
        ttext22= ''.join(filter(lambda x: x.isalpha() or x.isspace(), ttext21))

        #print(ttext)


        translated_text1 = translation(ttext12,max_length=2000)[0]['translation_text']
        translated_text2 = translation(ttext22,max_length=2000)[0]['translation_text']
        #print(translated_text)'''

        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


        if(numlines>len(text)):
            numlines=len(text)
        lines = {"numlines":numlines}

        summary=""
        for i in range(lines['numlines']):
            sentence=ranked_sentences[i][1] + '.'
            summary+=sentence

        length=len(summary)
        #print(summary)

        chunks=summary.split(".")
        chunks.pop()
        #print(chunks)

        '''i = 0
        n =100
        chunks = []

        while i < length:
            if i+n < length:
                chunks.append(summary[i:i+n])
            else:
                chunks.append(summary[i:length])
            i += n'''


        i=0
        translated_text =""
        #tokenized_text_0 = tokenizer.prepare_seq2seq_batch([summary], return_tensors='pt')
        #translation_0 = model.generate(**tokenized_text_0)

        #while i<len(chunks):
        if(language=="French"):
            translated_text=frontend_freeng(summary)
            print(translated_text)
        elif(language=="Dutch"):
            text = ' '.join(chunks)
            #print(text)
            gs = goslate.Goslate()
            translated_text = translated_text + gs.translate(text,'nl')
            #print(translated_text)
        else:
            while i<len(chunks):
                if(language=="German"):
                    translated_text += translation1(chunks[i]+'.',max_length=1000)[0]['translation_text']
                    i+=1
                elif(language=="Romanian"):
                    translated_text += translation3(chunks[i]+'.',max_length=1000)[0]['translation_text']
                    i+=1
                elif(language=="Hindi"):
                    tokenized_text_0 = tokenizer.prepare_seq2seq_batch(chunks[i]+'.', return_tensors='pt')
                    translation_0 = model.generate(**tokenized_text_0)
                    translated_text += tokenizer.batch_decode(translation_0, skip_special_tokens=True)[0]
                    i+=1


        messages = {"video":video,"numlines":numlines,"language":language,"transcript":text1, "rsent":summary,"numoflines":len(text),"tsent":translated_text}

        return render_template('output2.html',messages=messages)

    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'Error: {err_msg}', category='danger')
    return render_template('entry2.html', form=form)

'''def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new'''

def laugh_removal(mytext):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  sentence_list = tokenizer.tokenize(mytext)
  #print(len(sentence_list))
  pattern = "(Laughter)"
  #pattern2 ="(Applause)"
  res = [sentence_list.index(i) for i in sentence_list if pattern in i ] #sorted list
  for j in res:
    sentence_list.pop(j-1)
    sentence_list.pop(j-1)
  return sentence_list



@app.route('/entry1',methods=['GET','POST'])
def entry_page():
    form = EntryForm()
    if form.validate_on_submit():
        vid = form.videoid.data
        numlines = form.numlines.data
        language = form.language.data
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            text = [x['text'] for x in transcript]


        except:
            text=['!!!! Transcript Not Available for specified ID. !!!!']
            numlines=0

        if(int(numlines)>len(text)):
            numlines=len(text)

        mytext = [" ".join(t.split("\n")) for t in text]
        final_text = " ".join(mytext)
        text= laugh_removal(final_text)
        word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs

        f.close()
        # remove punctuations, numbers and special characters
        #clean_sentences = pd.Series(text).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in text]


        #stop_words = stopwords.words('english')
        #clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
            sim_mat = np.zeros([len(text), len(text)])
        for i in range(len(text)):
            for j in range(len(text)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(text)), reverse=True)
        #ranked_sentences = list(set(ranked_sentences)
        '''ttext1=ranked_sentences[:numlines//2]
        ttext2=ranked_sentences[((numlines//2)+1):numlines]

        ttext11= str(ttext1).strip('[]')
        ttext12= ''.join(filter(lambda x: x.isalpha() or x.isspace(), ttext11))

        ttext21= str(ttext2).strip('[]')
        ttext22= ''.join(filter(lambda x: x.isalpha() or x.isspace(), ttext21))

        #print(ttext)


        translated_text1 = translation(ttext12,max_length=2000)[0]['translation_text']
        translated_text2 = translation(ttext22,max_length=2000)[0]['translation_text']
        #print(translated_text)'''
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


        if(numlines>len(text)):
            numlines=len(text)
        lines = {"numlines":numlines}
        summary=""
        for i in range(lines['numlines']):
            summary+=ranked_sentences[i][1]

        length=len(summary)
        #print(summary)

        chunks=summary.split(".")
        chunks.pop()
        #print(chunks)

        '''i = 0
        n =100
        chunks = []

        while i < length:
            if i+n < length:
                chunks.append(summary[i:i+n])
            else:
                chunks.append(summary[i:length])
            i += n'''


        i=0
        translated_text =""
        #tokenized_text_0 = tokenizer.prepare_seq2seq_batch([summary], return_tensors='pt')
        #translation_0 = model.generate(**tokenized_text_0)
        if(language=="French"):
            translated_text=frontend_freeng(summary)
            print(translated_text)
        elif(language=="Dutch"):
            text = ' '.join(chunks)
            #print(text)
            gs = goslate.Goslate()
            translated_text = translated_text + gs.translate(text,'nl')
        else:
            while i<len(chunks):
                if(language=="German"):
                    translated_text += translation1(chunks[i]+'.',max_length=1000)[0]['translation_text']
                    i+=1
                elif(language=="Romanian"):
                    translated_text += translation3(chunks[i]+'.',max_length=1000)[0]['translation_text']
                    i+=1
                elif(language=="Hindi"):
                    tokenized_text_0 = tokenizer.prepare_seq2seq_batch(chunks[i]+'.', return_tensors='pt')
                    translation_0 = model.generate(**tokenized_text_0)
                    translated_text += tokenizer.batch_decode(translation_0, skip_special_tokens=True)[0]
                    i+=1

        messages = {"videoid":vid, "numlines":numlines,"language":language,"transcript":final_text, "rsent":ranked_sentences,"numoflines":len(text),"tsent":translated_text}

        return render_template('output.html',messages=messages)

    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'Error: {err_msg}', category='danger')
    return render_template('entry1.html', form=form)

'''def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new'''

def laugh_removal(mytext):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  sentence_list = tokenizer.tokenize(mytext)
  #print(len(sentence_list))
  pattern = "(Laughter)"
  #pattern2 ="(Applause)"
  res = [sentence_list.index(i) for i in sentence_list if pattern in i ] #sorted list
  for j in res:
    sentence_list.pop(j-1)
    sentence_list.pop(j-1)
  return sentence_list
