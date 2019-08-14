import json
import pandas as pd
import nltk
import re
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer 


## functions:
# 1. collect textual_cloze data from raw data
def extract_textual_cloze_data(rawFile='./data/train.json', task='textual_cloze'):
    f = open(rawFile, 'r', encoding='utf8').read()
    task_dict = json.loads(f)
    data = task_dict['data']#train file format {'data':[{},{},{}]} 
    recipes = [] 
    for recipe in data:
        if recipe['task'] == task:
            recipes.append(recipe) 
    new_data={}
    new_data['data'] = recipes 
    return new_data

# 2. clean the data: 
# 2.1 remove special symbol
def remove_special_symbol(text):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 
    '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', 
    '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 
    '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã',
    '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
    '≤', '‡', '√', ]
    text = str(text)
    for punct in puncts:
        text = text.replace(punct, '')
    return text

# 2.2 segmentation
def segmentation(text):
    text = str(text)
    return text.split()

# 2.3 remove stopword
def remove_stopword(splited_sentence):
    stopwords_set = set(stopwords.words('english')) 
    filter_sentence= [w for w in splited_sentence if w not in stopwords_set]
    return filter_sentence

# 2.4 Normalization: stemming and lemmatization
# 2.4.1 Stemming:
def stemming(splited_sentence):
    stemmer = SnowballStemmer("english") # choose language
    return [stemmer.stem(w) for w in splited_sentence]

# 2.4.2 Lemmatization
def lemmatization(splited_sentence):
    wnl = WordNetLemmatizer() 
    return [wnl.lemmatize(w) for w in splited_sentence]

# 2.5 lower case to reduce the number of the words
def lower_case(splited_sentence):
    return [w.lower() for w in splited_sentence]

# 2.6 correct mis-spelling
mispell_dict = {"aren't" : "are not", 
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying",
"Aren't" : "are not",
"Can't" : "cannot",
"Couldn't" : "could not",
"Didn't" : "did not",
"Doesn't" : "does not",
"Don't" : "do not",
"Hadn't" : "had not",
"Hasn't" : "has not",
"Haven't" : "have not",
"He'd" : "he would",
"He'll" : "he will",
"He's" : "he is",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"Isn't" : "is not",
"It's" : "it is",
"It'll":"it will",
"I've" : "I have",
"Let's" : "let us",
"Mightn't" : "might not",
"Mustn't" : "must not",
"Shan't" : "shall not",
"She'd" : "she would",
"She'll" : "she will",
"She's" : "she is",
"Shouldn't" : "should not",
"That's" : "that is",
"There's" : "there is",
"They'd" : "they would",
"They'll" : "they will",
"They're" : "they are",
"They've" : "they have",
"We'd" : "we would",
"We're" : "we are",
"Weren't" : "were not",
"We've" : "we have",
"What'll" : "what will",
"What're" : "what are",
"What's" : "what is",
"What've" : "what have",
"Where's" : "where is",
"Who'd" : "who would",
"Who'll" : "who will",
"Who're" : "who are",
"Who's" : "who is",
"Who've" : "who have",
"Won't" : "will not",
"Wouldn't" : "would not",
"You'd" : "you would",
"You'll" : "you will",
"You're" : "you are",
"You've" : "you have",
"Wasn't": "was not",
"We'll":" will",
"Didn't": "did not",
"'s":"is"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def extract_num_letter(string):
    return re.sub(r"[^a-zA-Z0-9]"," ",string)

# 2 clean string
def clean_string(string):
    cleaned_str = replace_typical_misspell(string)
    cleaned_str = extract_num_letter(cleaned_str)
    cleaned_str = remove_special_symbol(cleaned_str)
    cleaned_str = segmentation(cleaned_str)
    # cleaned_str = remove_stopword(cleaned_str)
    # cleaned_str = stemming(cleaned_str)
    # cleaned_str = lemmatization(cleaned_str)
    cleaned_str = lower_case(cleaned_str)
    return cleaned_str

# 4.load the raw data, preprocess, output the cleaned data
def preprocess(cleanFile, rawFile='./data/train.json', task='textual_cloze', structure='hierarchy'): #, imageFeatureFile='../data/training_features_resnet50.json'):
    try: # if we alread have the cleaned file
        # load cleaned data
        f = open(cleanFile, 'r', encoding='utf8')
        data = f.read()
        recipe = json.loads(data) #json file contains data in str, convert str to dict
        recipe_id = recipe['id']
        recipe_text = recipe['text']
        recipe_answer = recipe['answer']
        recipe_choice = recipe['choice']
        recipe_question = recipe['question']
        recipe_image = recipe['image']
        f.close()
        return recipe_text, recipe_image, recipe_question, recipe_choice, recipe_answer,recipe_id
    except IOError: #File is not accessible, create a clean file
        textual_cloze_data = extract_textual_cloze_data(rawFile, task)
        #img_features = read_imgs_file(imageFeatureFile)
        data = textual_cloze_data['data']
        question_list = []
        choice_list = []
        id_list=[]
        answer_list = []
        text_list = []
        image_list = []
        if structure == 'entirety':
            for recipe in data:
                text_entire = ''
                question_entire = ''
                choices = []
                images = [] # add image data here
                for step in recipe['context']:
                    text_entire = text_entire  + ' ' + step['body']
                    step_imgs = step['images']
                    images.append(step_imgs)
                for step_str in recipe['question']:
                    question_entire = question_entire + ' ' + step_str
                for choice in recipe['choice_list']:
                    choices.append(clean_string(choice))
                question_list.append(clean_string(question_entire))
                text_list.append(clean_string(text_entire))
                image_list.append(images)
                choice_list.append(choices)
                id_list.append(recipe['recipe_id'])
                answer_list.append(recipe['answer'])
        elif structure == 'hierarchy':
            for recipe in data:
                texts = []
                questions = []
                choices = []
                images = []
                for step in recipe['context']:
                    texts.append(clean_string(step['body']))
                    step_imgs = step['images']
                    for img in step_imgs:
                        images.append(img)
                for step in recipe['question']:
                    questions.append(clean_string(step))
                for choice in recipe['choice_list']:
                    choices.append(clean_string(choice))
                question_list.append(questions)
                text_list.append(texts)
                image_list.append(images)
                choice_list.append(choices)
                id_list.append(recipe['recipe_id'])
                answer_list.append(recipe['answer'])
        recipes = {}
        recipes['text'] = text_list
        recipes['id']=id_list
        recipes['answer'] = answer_list
        recipes['choice'] = choice_list
        recipes['question'] = question_list
        recipes['image'] = image_list
        with open(cleanFile, 'w', encoding='utf8') as f:
            json.dump(recipes, f, indent=4, ensure_ascii=False) # convert dict to str and write, indent means change row
        return text_list, image_list, question_list, choice_list, answer_list,id_list

if __name__ == "__main__":
    #preprocess(cleanFile='./data/entirety/train_cleaned.json', rawFile='./data/train.json', task='textual_cloze', structure='entirety')
    #preprocess(cleanFile='./data/entirety/val_cleaned.json', rawFile='./data/val.json', task='textual_cloze', structure='entirety')
    #preprocess(cleanFile='../data/hierarchy/train_cleaned.json', rawFile='../data/train.json', task='textual_cloze', structure='hierarchy')#, imageFeatureFile='../data/training_features_resnet50.json')
    #preprocess(cleanFile='../data/hierarchy/val_cleaned.json', rawFile='../data/val.json', task='textual_cloze', structure='hierarchy')# ,  imageFeatureFile='../data/validation_features_resnet50.json')
    preprocess(cleanFile='../data/hierarchy/test_cleaned.json', rawFile='../data/test.json', task='textual_cloze', structure='hierarchy') #, imageFeatureFile='../data/test_features_resnet50.json')
