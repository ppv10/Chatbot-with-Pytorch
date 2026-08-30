import json
import random
from pathlib import Path

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#nltk.download('wordnet')
#nltk.download('punkt_tab')

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = Path(intents_path)

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None
        self._vocabulary_index = {}

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words
    
    def bag_of_words(self, words):
        if len(self._vocabulary_index) != len(self.vocabulary):
            self._vocabulary_index = {
                word: index for index, word in enumerate(self.vocabulary)
            }

        bag = [0] * len(self.vocabulary)
        for word in set(words):
            index = self._vocabulary_index.get(word)
            if index is not None:
                bag[index] = 1

        return bag

    def parse_intents(self):
        if not self.intents_path.is_file():
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")

        with self.intents_path.open(encoding='utf-8') as file:
            intents_data = json.load(file)

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))

        if not self.documents:
            raise ValueError('The intents file does not contain any training patterns.')

        self.vocabulary = sorted(set(self.vocabulary))
        self._vocabulary_index = {
            word: index for index, word in enumerate(self.vocabulary)
        }
    
    def prepare_data(self):
        if not self.documents:
            raise RuntimeError('Call parse_intents() before preparing training data.')

        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intents_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intents_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        if self.X is None or self.y is None:
            raise RuntimeError('Call prepare_data() before training the model.')

        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = .0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        if self.model is None:
            raise RuntimeError('Train a model before saving it.')

        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r', encoding='utf-8') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=True)
        )
        self.model.eval()

    def process_message(self, input_message):
        if self.model is None:
            raise RuntimeError('Load or train a model before processing messages.')

        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()
        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None

def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'AMD', 'INTC']

    print(random.sample(stocks, 3))

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent

    """assistant = ChatbotAssistant('intents.json', function_mappings= {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=.001, epochs=100)
    
    assistant.save_model('chatbot_model.pth', 'dimensions.json')"""

    assistant = ChatbotAssistant(project_dir / 'intents.json', function_mappings={'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model(project_dir / 'chatbot_model.pth', project_dir / 'dimensions.json')

    while True:
        message = input('Enter your message: (type /quit to quit)')

        if message == '/quit':
            break

        print(assistant.process_message(message))
