# Chatbot with PyTorch

A small intent-classification chatbot built with PyTorch and NLTK, inspired by
NeuralNine.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"
python main.py
```

The committed `chatbot_model.pth` and `dimensions.json` are used for inference.
To retrain the model, uncomment the training block in `main.py`, run it once,
then restore the inference block.
