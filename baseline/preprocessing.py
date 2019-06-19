import json
import nltk
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

# 2 clean string
def clean_string(string):
    cleaned_str = remove_special_symbol(string)
    cleaned_str = segmentation(cleaned_str)
    cleaned_str = remove_stopword(cleaned_str)
    # cleaned_str = stemming(cleaned_str)
    cleaned_str = lemmatization(cleaned_str)
    cleaned_str = lower_case(cleaned_str)
    return cleaned_str


# 3. load the raw data, preprocess, output the cleaned data
def preprocess(cleanFile, rawFile='./data/train.json', task='textual_cloze', structure='entirety'):
    try: # if we alread have the cleaned file
        # load cleaned data
        f = open(cleanFile, 'r', encoding='utf8')
        data = f.read()
        recipe = json.loads(data) #json file contains data in str, convert str to dict
        recipe_text = recipe['text']
        recipe_answer = recipe['answer']
        recipe_choice = recipe['choice']
        recipe_question = recipe['question']
        recipe_image = recipe['image']
        f.close()
        return recipe_text, recipe_image, recipe_question, recipe_choice, recipe_answer
    except IOError: #File is not accessible, create a clean file
        textual_cloze_data = extract_textual_cloze_data(rawFile, task)
        data = textual_cloze_data['data']
        question_list = []
        choice_list = []
        answer_list = []
        text_list = []
        image_list = []
        if structure == 'entirety':
            for recipe in data:
                text_entire = ''
                question_entire = ''
                choices = []
                images = []
                for step in recipe['context']:
                    text_entire = text_entire  + ' ' + step['body']
                    images.append(step['images'])
                for step_str in recipe['question']:
                    question_entire = question_entire + ' ' + step_str
                for choice in recipe['choice_list']:
                    choices.append(clean_string(choice))
                question_list.append(clean_string(question_entire))
                text_list.append(clean_string(text_entire))
                image_list.append(images)
                choice_list.append(choices)
                answer_list.append(recipe['answer'])
        elif structure == 'hierarchy':
            for recipe in data:
                texts = []
                questions = []
                choices = []
                images = []
                for step in recipe['context']:
                    texts.append(clean_string(step['body']))
                    images.append(step['images'])
                for step in recipe['question']:
                    questions.append(clean_string(step))
                for choice in recipe['choice_list']:
                    choices.append(clean_string(choice))
                question_list.append(questions)
                text_list.append(texts)
                image_list.append(images)
                choice_list.append(choices)
                answer_list.append(recipe['answer'])
        recipes = {}
        recipes['text'] = text_list
        recipes['answer'] = answer_list
        recipes['choice'] = choice_list
        recipes['question'] = question_list
        recipes['image'] = image_list
        with open(cleanFile, 'w', encoding='utf8') as f:
            json.dump(recipes, f, indent=4, ensure_ascii=False) # convert dict to str and write, indent means change row
        return text_list, image_list, question_list, choice_list, answer_list



if __name__ == "__main__":
    #preprocess(cleanFile='./data/entirety/train_cleaned.json', rawFile='./data/train.json', task='textual_cloze', structure='entirety')
    #preprocess(cleanFile='./data/entirety/val_cleaned.json', rawFile='./data/val.json', task='textual_cloze', structure='entirety')
    preprocess(cleanFile='./data/hierarchy/train_cleaned.json', rawFile='./data/train.json', task='textual_cloze', structure='hierarchy')
    preprocess(cleanFile='./data/hierarchy/val_cleaned.json', rawFile='./data/val.json', task='textual_cloze', structure='hierarchy')