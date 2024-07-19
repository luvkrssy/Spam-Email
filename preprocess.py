{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "051f59d1-657d-4e48-88b6-095a8bd6d0d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\User\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import string\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "nltk.download('stopwords')\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def preprocess_text(text):\n",
    "    # Convert to lowercase\n",
    "    text = text.lower()\n",
    "    # Remove punctuation\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
    "    # Remove stopwords\n",
    "    text = ' '.join([word for word in text.split() if word not in stop_words])\n",
    "    return text\n",
    "\n",
    "def load_and_preprocess_data(filepath):\n",
    "    data = pd.read_csv(filepath)\n",
    "    data['processed_text'] = data['text'].apply(preprocess_text)\n",
    "    X = data['processed_text']\n",
    "    y = data['label']\n",
    "    return train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
