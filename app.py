import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import re
import string
import abbreviations as abb

# Set page config
st.set_page_config(
    page_title="Vietnamese Comment Hate Detection",
    layout="centered"
)

# Custom model class
class CommentClassifier(nn.Module):
    def __init__(self, phobert, n_classes):
        super(CommentClassifier, self).__init__()
        self.bert = phobert#AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Load PhoBERT model and tokenizer
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Initialize classifier
    model = CommentClassifier(phobert, 3)
    
    # # Load trained weights
    # state_dict = torch.load('model/phobert.pth', map_location=torch.device('cpu'))
    # # Remove unexpected keys
    # state_dict = {k: v for k, v in state_dict.items() if k != "bert.embeddings.position_ids"}
    # model.load_state_dict(state_dict, strict=False)
    # model.eval()

    model.load_state_dict(torch.load('model/phobert.pth'))
    
    return model, tokenizer

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\[\]\(\)\{\}]", " ", text)
    text = re.sub(r"[\'\.\,\-\:\"!~\?\*\']", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Create an instance of Abb class and use it to process the text
    abb_processor = abb.Abb()
    processed_words = []
    for word in text.split():
        processed_word = abb_processor.rep(word)
        processed_words.append(processed_word)
    
    return ' '.join(processed_words)

# Main app
def main():
    st.title("Vietnamese Text Classification")
    st.write("Classify Vietnamese text into: Neutral (0), Offensive (1), or Hate (2)")
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Text input
    text_input = st.text_area("Enter Vietnamese text to classify:", height=150)
    
    if st.button("Classify"):
        if text_input:
            # Preprocess text
            processed_text = preprocess_text(text_input)
            
            # Tokenize
            inputs = tokenizer.encode_plus(
                processed_text,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Get prediction
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()

            
            # Display results
            st.subheader("Classification Result:")
            labels = ["Neutral", "Offensive", "Hate"]
            st.write(f"Predicted class: {labels[prediction]} ({prediction})")
            
            # Display probabilities
            st.subheader("Class Probabilities:")
            for i, prob in enumerate(probabilities[0]):
                st.write(f"{labels[i]}: {prob.item():.2%}")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main() 