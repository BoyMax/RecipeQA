import gensim
from gensim.models.doc2vec import Doc2Vec
from preprocessing import preprocess
from gensim.test.utils import get_tmpfile

def get_text(cleanFile, rawFile, task, structure):
    text_list, image_list, question_list, choice_list, answer_list = preprocess(cleanFile, rawFile, task, structure)
    return text_list, question_list

def train_text(texts):
    train_X = []
    TaggedDocument = gensim.models.doc2vec.TaggedDocument
    for i, text in enumerate(texts):
        doc = TaggedDocument(text, tags=[i])
        train_X.append(doc)
    return train_X

def train(train_X, size=100):
    model = Doc2Vec(train_X, min_count=1, window=3, vector_size=size, sample=1e-3, negative=5, workers=4)
    model.train(train_X, total_examples=model.corpus_count, epochs=50)
    return model

def save_model(model, fname = "doc2vec_text_model"):
    #fname = get_tmpfile(fname)
    model.save(fname)
    return model

def load_model(fname):
    model = Doc2Vec.load(fname)
    return model

if __name__ == "__main__":
    texts, questions = get_text(cleanFile='../data/entirety/train_cleaned.json', rawFile='../data/train.json', task='textual_cloze', structure='entirety')
    train_X = train_text(texts)
    model = train(train_X, size=100)
    save_model(model, "doc2vec_text_model")
    