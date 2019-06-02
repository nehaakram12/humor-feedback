import flask
from flask import request, jsonify
import sys
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#initializing flask api
app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def init():
    print("Please wait while the Data-Models load!...")


@app.route('/', methods=['GET'])
def home():
    return '''<h1>S.A.R.C.A.S.M.A.N.I.A</h1>
<p>Can you BEEEEEEEE more sarcastic??!!.</p><p>A prototype API for sarcasmania.</p>'''

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/api/sarcasmania', methods=['GET'])
def api_text():
    inputsen=""
    #taking sentence from url params
    if 'text' in request.args:
        inputsen = (request.args['text'])
    else:
        return "Error: No text field provided. Please specify text."
    print("Input Line: ", inputsen)
    humorlabel=0
    #taking score label from url params
    if 'label' in request.args:
        humorlabel = (request.args['label'])
    else:
        return "Error: No label field provided. Please specify label."
    print("Humor label: ", humorlabel)

    d = []
    #opening file
    dataFile = open('output1.txt', 'rb')
    #loading file
    d = pickle.load(dataFile)
    #loading the previously saved model
    filename = 'partial_fit_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    #creating features of data
    X = create_tfidf_training_data(d, inputsen)
    
    #retraining the model on the new data
    loaded_model.partial_fit(X, [humorlabel])

    # saving the updated model back to model file
    filename = open('partial_fit_model.sav', 'wb')
    pickle.dump(loaded_model, filename)
    dataFile.close()
    filename.close()
    print("Feedback updated successfully into the model")

    results = {
     'Input': inputsen,
     'Humor Label': humorlabel,
    }

    return jsonify(results)


def create_tfidf_training_data(docs,column):
    y = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    t=vectorizer.transform([column])

    return t


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print("* Loading model and Flask starting server...please wait until server has fully started")
    init()
    app.run(threaded=True)
