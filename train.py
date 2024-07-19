from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def train_model(X_train, y_train, model_type='logistic'):
    if model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = SVC()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model
