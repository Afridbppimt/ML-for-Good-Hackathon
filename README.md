#Overview of The Project

This is a NLP based project focused on open source surveyed data where people demonstrated their views based on the experience during COVID-19 from individual or family level.
The effort primarily focus on understanding the relation behind all the aspects mentioned by people as an experience & how that experience is 
leading to a conclussion to mental health or stress or changes in daily activity.

With regrdas to that we built a rich Knowledge Tree following the concept of relation between User's motive related to any concept or action.
Intention is to use that as a Knowledge Base  for any kind of relation & getting answer to it.
We have focused on creating dynamic FAQ generation from unstructured text with Para Phrasing Capability that can be used to reach to an answer very easily.
Later we can use that FAQ to get different type of answer from the knowledge Base to get answer & related answer as well.

We found the different component like mental stress or sleep quality or other feature like that are closely related to anyone mental's health collected from survey report.
To address that we have built a classification model to target those feature which is playing key role to degrage anyone's mental condition & target that to help them recover.
We could use that feature to recommend any suggestion based on other's people action from Knowledge base.

### Currently Supported Capabilities :

### Currently Supported Question Generation Capabilities :
<pre>
1. Knowledge Base Engine
1. Multiple Choice Questions (MCQs)
2. Boolean Questions (Yes/No)
3. Paraphrasing any Question
4. Mental Worriedness Classification Model
5. Keyword Extraction
</pre>

## Installation Steps

### Knowledge_Graph Libraries

1. Spacy - 2.2.4
2. Pandas - 1.1.5
3. Networkx - 2.6.3
4. Matplotlib - 3.2.2
5. sklearn - 1.0.1

### Data Used : crisislogger.csv

### Running Code :

<pre>
1. Read Text Paragraphs
1. Break the sentences using Engish Cojugation Rules using Regex
2. Find Entities Like Subject & Object of the phrase or sentence
3. Find relation between Subject & Object
4. Convert the above 3 parameters to organized data
5. Plot the Knowledge Graph
</pre>

![Kg Mental Health](/Images/KG_Mental_health.png)

### 1.1 FAQ Generation Libraries
```
https://pypi.org/project/sense2vec/
https://boudinfl.github.io/pke/build/html/index.html

pip install -r requirements.txt

pip install git+https://github.com/boudinfl/pke.git

python -m nltk.downloader universal_tagset
python -m spacy download en 
```
### 1.2 Download and extract zip of Sense2vec wordvectors that are used for generation of multiple choices.
```
wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
```

## 2. Running the code

### 2.1 Generate boolean (Yes/No) Questions
```
	##Read the CrisisLogger to dataframe
	fields = ['upload_id', 'transcriptions']
	logger_df = pd.read_csv(r'C:\Users\sanje\Desktop\ML-for-Good-Hackathon-main\ML-for-Good-Hackathon-main\Data\CrisisLogger\crisislogger.csv', usecols=fields)
	print('Transcript: {}'.format(logger_df.transcriptions.iloc[24]))
	payload = {
				"input_text": logger_df.transcriptions.iloc[24]
	}

	##Generating Boolean Questions from above Transcripts
	output = qe.predict_boolq(payload)
	#pprint (output)

	boolian_questions_df = pd.DataFrame(output.items(), columns=['Type', 'Data'])
	boolian_df = pd.DataFrame(boolian_questions_df['Data'].iloc[2], columns=['BooleanQuestions']) 
	display(HTML(boolian_df.to_html()))

```

<details>
<summary>Show Output</summary>

Transcript: I’m a woman, aged 63 (almost 64) and still working full time because of the change in the pension age. I was hospitalised with Swine flu in 2009 and it had a big effect on my health. I was diagnosed with asthma in 2010. My lungs have never fully recovered from the severe case of Swine flu. 
I normally work nine to five and there’s have been times over the past two years when I’ve felt very tired and weary. But it wasn’t until this pandemic hit that I actually felt old for the first time in my life. I’ll admit that I went into a blind spin panic when I realised how bad this pandemic was at the end of March. Stress being one of the triggers for my asthma, the feeling of dread and tightness in my chest caused my asthma to peak. I remembered how ill i was in 2009 and the feeling that I just wanted to die and due to my weakened lungs I couldn’t shake the fear. I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies. My son went to pick it up from the pharmacy the next day, only to discover they couldn’t get supplies of my usual medication. Another call to my doctor and she prescribed an alternative to the medication I had been on since 2010 and my initial asthma diagnosis. A few days later my wheezing had eased but I found I was suffering from severe reflux at nights. So bad it almost choked me on a few occasions, thus adding to my considerable stress. Another call to my doctor, who said the reflux was caused by stress so I was given acid suppressants and signed off work until everything could settle down a bit. I work for a train operating company so I am considered a key worker. 
In summary, I have found the last six weeks to be the most stressful weeks of my life. I’ve always considered myself to be as fit and able as my younger colleagues but this crisis has made me realise I need to take better care of myself and not put my job first. This virus has reminded me that I am getting old.

```

{'Boolean Questions': ['Is there a cure for the swine flu?',
                       'Is there a cure for swine flu?',
                       'Is there such a thing as swine flu?'],
}


BooleanQuestions
0	Is there a cure for the swine flu?
1	Is there a cure for swine flu?
2	Is there such a thing as swine flu?

```
</details>

### 2.2 Generate Paraphrasing Questions
```
	##Generating Paraphrased Questions from above Transcripts
	qg = QGen()
	output = qg.paraphrase(payload)
	#pprint (output)

	paraphrase_questions_df = pd.DataFrame(output.items(), columns=['Type', 'Data'])
	paraphrase_df = pd.DataFrame(paraphrase_questions_df['Data'].iloc[2], columns=['ParaphrasedTarget']) 
	display(HTML(paraphrase_df.to_html()))
    
```

<details>
<summary>Show Output</summary>
            
```
{'Count': 3,
 'Paraphrased Questions': ["ParaphrasedTarget: I'm a woman, aged 63 (almost "
                           '64) and still working full time because of the '
                           'change in the pension age. I was hospitalised with '
                           'Swine flu in 2009 and it had',
                           'ParaphrasedTarget: I am a woman, aged 63 (almost '
                           '64) and still working full time because of the '
                           'change in the pension age. I was hospitalised with '
                           'Swine flu in 2009 and it had ',
                           "ParaphrasedTarget: I'm a woman, 63 (almost 64) and "
                           'still working full time because of the change in '
                           'the pension age. I was hospitalised with Swine flu '
                           'in 2009 and it had '],
 'Question': 'I’m a woman, aged 63 (almost 64) and still working full time '
             'because of the change in the pension age. I was hospitalised '
             'with Swine flu in 2009 and it had a big effect on my health. I '
             'was diagnosed with asthma in 2010. My lungs have never fully '
             'recovered from the severe case of Swine flu. \n'
             'I normally work nine to five and there’s have been times over '
             'the past two years when I’ve felt very tired and weary. But it '
             'wasn’t until this pandemic hit that I actually felt old for the '
             'first time in my life. I’ll admit that I went into a blind spin '
             'panic when I realised how bad this pandemic was at the end of '
             'March. Stress being one of the triggers for my asthma, the '
             'feeling of dread and tightness in my chest caused my asthma to '
             'peak. I remembered how ill i was in 2009 and the feeling that I '
             'just wanted to die and due to my weakened lungs I couldn’t shake '
             'the fear. I rang my doctor (not wanting to visit the surgery) '
             'and she advised I double up my asthma medication and issued a '
             'prescription for extra supplies. My son went to pick it up from '
             'the pharmacy the next day, only to discover they couldn’t get '
             'supplies of my usual medication. Another call to my doctor and '
             'she prescribed an alternative to the medication I had been on '
             'since 2010 and my initial asthma diagnosis. A few days later my '
             'wheezing had eased but I found I was suffering from severe '
             'reflux at nights. So bad it almost choked me on a few occasions, '
             'thus adding to my considerable stress. Another call to my '
             'doctor, who said the reflux was caused by stress so I was given '
             'acid suppressants and signed off work until everything could '
             'settle down a bit. I work for a train operating company so I am '
             'considered a key worker. \n'
             'In summary, I have found the last six weeks to be the most '
             'stressful weeks of my life. I’ve always considered myself to be '
             'as fit and able as my younger colleagues but this crisis has '
             'made me realise I need to take better care of myself and not put '
             'my job first. This virus has reminded me that I am getting old.'}
			 
    ParaphrasedTarget
0	ParaphrasedTarget: I'm a woman, aged 63 (almost 64) and still working full time because of the change in the pension age. I was hospitalised with Swine flu in 2009 and it had
1	ParaphrasedTarget: I am a woman, aged 63 (almost 64) and still working full time because of the change in the pension age. I was hospitalised with Swine flu in 2009 and it had
2	ParaphrasedTarget: I'm a woman, 63 (almost 64) and still working full time because of the change in the pension age. I was hospitalised with Swine flu in 2009 and it had
```
</details> 


### 2.3 Generate FAQ Questions

```
##Generate FAQ's and ther answers
output = qg.predict_shortq(payload)
#pprint (output)

faq_questions_df = pd.DataFrame(output.items(), columns=['Type', 'Data'])
#display(HTML(faq_questions_df.to_html()))
##FAQ's DF
faq_df = pd.DataFrame(faq_questions_df['Data'].iloc[1]) 
display(HTML(faq_df.to_html()))
```


<details>
<summary>Show Output</summary>

 ```
 {'questions': [{'Answer': 'asthma medication',
                'Question': 'What did my doctor tell me to double up on?',
                'context': 'I rang my doctor (not wanting to visit the '
                           'surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies.',
                'id': 1},
               {'Answer': 'reflux',
                'Question': 'What was the cause of my wheezing?',
                'context': 'Another call to my doctor, who said the reflux was '
                           'caused by stress so I was given acid suppressants '
                           'and signed off work until everything could settle '
                           'down a bit. A few days later my wheezing had eased '
                           'but I found I was suffering from severe reflux at '
                           'nights.',
                'id': 2},
               {'Answer': 'supplies',
                'Question': 'What did my doctor tell me to double up my asthma '
                            'medication and issue a prescription for?',
                'context': 'I rang my doctor (not wanting to visit the '
                           'surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies. My son went to pick it up from the '
                           'pharmacy the next day, only to discover they '
                           'couldn’t get supplies of my usual medication.',
                'id': 3},
               {'Answer': 'doctor',
                'Question': 'Who advised me to double up my asthma medication '
                            'and issued a prescription for extra supplies?',
                'context': 'Another call to my doctor, who said the reflux was '
                           'caused by stress so I was given acid suppressants '
                           'and signed off work until everything could settle '
                           'down a bit. I rang my doctor (not wanting to visit '
                           'the surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies. Another call to my doctor and she '
                           'prescribed an alternative to the medication I had '
                           'been on since 2010 and my initial asthma '
                           'diagnosis.',
                'id': 4}],
 'statement': 'I’m a woman, aged 63 (almost 64) and still working full time '
              'because of the change in the pension age. I was hospitalised '
              'with Swine flu in 2009 and it had a big effect on my health. I '
              'was diagnosed with asthma in 2010. My lungs have never fully '
              'recovered from the severe case of Swine flu. I normally work '
              'nine to five and there’s have been times over the past two '
              'years when I’ve felt very tired and weary. But it wasn’t until '
              'this pandemic hit that I actually felt old for the first time '
              'in my life. I’ll admit that I went into a blind spin panic when '
              'I realised how bad this pandemic was at the end of March. '
              'Stress being one of the triggers for my asthma, the feeling of '
              'dread and tightness in my chest caused my asthma to peak. I '
              'remembered how ill i was in 2009 and the feeling that I just '
              'wanted to die and due to my weakened lungs I couldn’t shake the '
              'fear. I rang my doctor (not wanting to visit the surgery) and '
              'she advised I double up my asthma medication and issued a '
              'prescription for extra supplies. My son went to pick it up from '
              'the pharmacy the next day, only to discover they couldn’t get '
              'supplies of my usual medication. Another call to my doctor and '
              'she prescribed an alternative to the medication I had been on '
              'since 2010 and my initial asthma diagnosis. A few days later my '
              'wheezing had eased but I found I was suffering from severe '
              'reflux at nights. So bad it almost choked me on a few '
              'occasions, thus adding to my considerable stress. Another call '
              'to my doctor, who said the reflux was caused by stress so I was '
              'given acid suppressants and signed off work until everything '
              'could settle down a bit. I work for a train operating company '
              'so I am considered a key worker. In summary, I have found the '
              'last six weeks to be the most stressful weeks of my life. I’ve '
              'always considered myself to be as fit and able as my younger '
              'colleagues but this crisis has made me realise I need to take '
              'better care of myself and not put my job first. This virus has '
              'reminded me that I am getting old.'}
			  
			  
Question	Answer	id	context
0	What did my doctor tell me to double up on?	asthma medication	1	I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies.
1	What was the cause of my wheezing?	reflux	2	Another call to my doctor, who said the reflux was caused by stress so I was given acid suppressants and signed off work until everything could settle down a bit. A few days later my wheezing had eased but I found I was suffering from severe reflux at nights.
2	What did my doctor tell me to double up my asthma medication and issue a prescription for?	supplies	3	I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies. My son went to pick it up from the pharmacy the next day, only to discover they couldn’t get supplies of my usual medication.
3	Who advised me to double up my asthma medication and issued a prescription for extra supplies?	doctor	4	Another call to my doctor, who said the reflux was caused by stress so I was given acid suppressants and signed off work until everything could settle down a bit. I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies. Another call to my doctor and she prescribed an alternative to the medication I had been on since 2010 and my initial asthma diagnosis.
 ```
</details>

### 2.4 Generate MCQs Questions
```
##Generate MCQ's and ther answers
output = qg.predict_mcq(payload)
#pprint (output)

mcq_questions_df = pd.DataFrame(output.items(), columns=['Type', 'Data'])
#display(HTML(faq_questions_df.to_html()))
##FAQ's DF
mcq_df = pd.DataFrame(mcq_questions_df['Data'].iloc[1]) 
display(HTML(mcq_df.to_html()))

```
<details>
<summary>Show Output</summary>
            
```
unning model for generation
 Sense2vec_distractors successful for word :  asthma medication
 Sense2vec_distractors successful for word :  reflux
 Sense2vec_distractors successful for word :  supplies
 Sense2vec_distractors successful for word :  doctor
{'questions': [{'answer': 'asthma medication',
                'context': 'I rang my doctor (not wanting to visit the '
                           'surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies.',
                'extra_options': ['Sudafed',
                                  'Own Medication',
                                  'Nasal Spray',
                                  'Antibiotics',
                                  'Aspirin',
                                  'Decongestants'],
                'id': 1,
                'options': ['Inhalers', 'Albuterol', 'Advair'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'What did my doctor tell me to double up '
                                      'on?',
                'question_type': 'MCQ'},
               {'answer': 'reflux',
                'context': 'Another call to my doctor, who said the reflux was '
                           'caused by stress so I was given acid suppressants '
                           'and signed off work until everything could settle '
                           'down a bit. A few days later my wheezing had eased '
                           'but I found I was suffering from severe reflux at '
                           'nights.',
                'extra_options': ['Nausea',
                                  'Heartburn',
                                  'Dry Mouth',
                                  'Ibs',
                                  'Gi Problems'],
                'id': 2,
                'options': ['Constipation', 'Gerd', 'Stomach Issues'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'What was the cause of my wheezing?',
                'question_type': 'MCQ'},
               {'answer': 'supplies',
                'context': 'I rang my doctor (not wanting to visit the '
                           'surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies. My son went to pick it up from the '
                           'pharmacy the next day, only to discover they '
                           'couldn’t get supplies of my usual medication.',
                'extra_options': [],
                'id': 3,
                'options': ['Rations', 'Building Materials', 'Stockpiles'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'What did my doctor tell me to double up '
                                      'my asthma medication and issue a '
                                      'prescription for?',
                'question_type': 'MCQ'},
               {'answer': 'doctor',
                'context': 'Another call to my doctor, who said the reflux was '
                           'caused by stress so I was given acid suppressants '
                           'and signed off work until everything could settle '
                           'down a bit. I rang my doctor (not wanting to visit '
                           'the surgery) and she advised I double up my asthma '
                           'medication and issued a prescription for extra '
                           'supplies. Another call to my doctor and she '
                           'prescribed an alternative to the medication I had '
                           'been on since 2010 and my initial asthma '
                           'diagnosis.',
                'extra_options': ['General Practitioner'],
                'id': 4,
                'options': ['Cardiologist', 'Doc', 'Primary Care Physician'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'Who advised me to double up my asthma '
                                      'medication and issued a prescription '
                                      'for extra supplies?',
                'question_type': 'MCQ'}],
 'statement': 'I’m a woman, aged 63 (almost 64) and still working full time '
              'because of the change in the pension age. I was hospitalised '
              'with Swine flu in 2009 and it had a big effect on my health. I '
              'was diagnosed with asthma in 2010. My lungs have never fully '
              'recovered from the severe case of Swine flu. I normally work '
              'nine to five and there’s have been times over the past two '
              'years when I’ve felt very tired and weary. But it wasn’t until '
              'this pandemic hit that I actually felt old for the first time '
              'in my life. I’ll admit that I went into a blind spin panic when '
              'I realised how bad this pandemic was at the end of March. '
              'Stress being one of the triggers for my asthma, the feeling of '
              'dread and tightness in my chest caused my asthma to peak. I '
              'remembered how ill i was in 2009 and the feeling that I just '
              'wanted to die and due to my weakened lungs I couldn’t shake the '
              'fear. I rang my doctor (not wanting to visit the surgery) and '
              'she advised I double up my asthma medication and issued a '
              'prescription for extra supplies. My son went to pick it up from '
              'the pharmacy the next day, only to discover they couldn’t get '
              'supplies of my usual medication. Another call to my doctor and '
              'she prescribed an alternative to the medication I had been on '
              'since 2010 and my initial asthma diagnosis. A few days later my '
              'wheezing had eased but I found I was suffering from severe '
              'reflux at nights. So bad it almost choked me on a few '
              'occasions, thus adding to my considerable stress. Another call '
              'to my doctor, who said the reflux was caused by stress so I was '
              'given acid suppressants and signed off work until everything '
              'could settle down a bit. I work for a train operating company '
              'so I am considered a key worker. In summary, I have found the '
              'last six weeks to be the most stressful weeks of my life. I’ve '
              'always considered myself to be as fit and able as my younger '
              'colleagues but this crisis has made me realise I need to take '
              'better care of myself and not put my job first. This virus has '
              'reminded me that I am getting old.',
 'time_taken': 5.615395545959473}
 
question_statement	question_type	answer	id	options	options_algorithm	extra_options	context
0	What did my doctor tell me to double up on?	MCQ	asthma medication	1	[Inhalers, Albuterol, Advair]	sense2vec	[Sudafed, Own Medication, Nasal Spray, Antibiotics, Aspirin, Decongestants]	I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies.
1	What was the cause of my wheezing?	MCQ	reflux	2	[Constipation, Gerd, Stomach Issues]	sense2vec	[Nausea, Heartburn, Dry Mouth, Ibs, Gi Problems]	Another call to my doctor, who said the reflux was caused by stress so I was given acid suppressants and signed off work until everything could settle down a bit. A few days later my wheezing had eased but I found I was suffering from severe reflux at nights.
2	What did my doctor tell me to double up my asthma medication and issue a prescription for?	MCQ	supplies	3	[Rations, Building Materials, Stockpiles]	sense2vec	[]	I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies. My son went to pick it up from the pharmacy the next day, only to discover they couldn’t get supplies of my usual medication.
3	Who advised me to double up my asthma medication and issued a prescription for extra supplies?	MCQ	doctor	4	[Cardiologist, Doc, Primary Care Physician]	sense2vec	[General Practitioner]	Another call to my doctor, who said the reflux was caused by stress so I was given acid suppressants and signed off work until everything could settle down a bit. I rang my doctor (not wanting to visit the surgery) and she advised I double up my asthma medication and issued a prescription for extra supplies. Another call to my doctor and she prescribed an alternative to the medication I had been on since 2010 and my initial asthma diagnosis.
```
</details>

### 2.5 Summary (Simple)
```
#Question Answering (Simple)
answer = AnswerPredictor()
output = answer.predict_answer(payload)
pprint (output)

```
<details>
<summary>Show Output</summary>
            
```
('I’m a woman, aged 63 (almost 64) and still working full time because of the '
 'change in the pension age. i was hospitalised with swine flu in 2009 and it '
 'had a big effect on my health. i was diagnosed with asthma in 2010. my lungs '
 'have never fully recovered from the severe case of swine flu. i normally '
 'work nine to five and there’s have been times over the past two years when '
 'i’ve felt very tired and weary. but it wasn’t until this pandemic hit')
```
</details>


### NLP models used

For maintaining meaningfulness in Questions, uses Three T5 models. One for Boolean Question generation, one for MCQs, FAQs, Paraphrasing and one for Summary answer generation.


# Mental Worriedness Classification Model
### Libraries
Pandas - 1.1.5

### Data Used : CRISIS_Adult_April_2020.csv, CRISIS_Adult_April_2021.csv, CRISIS_Adult_May_2020.csv, CRISIS_Adult_November_2020.csv

Model is to find out the best parmaters to describe the Adult's mental worriedness & classify the level of worriedness form individual level.

#### As a next step we want to take this project forward by combining the individual module so that we could find some valuable insights like source of Covid-19 or Mental Health and related symptomps from unstructured text.

