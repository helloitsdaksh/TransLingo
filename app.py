import spacy
import torch
import streamlit as st

from model import Seq2Seq, Encoder, Attention, Decoder
# Load spaCy models for German and English
de_nlp = spacy.load('de_core_news_sm')
en_nlp = spacy.load('en_core_web_sm')



# Example function to preprocess text using spaCy
def preprocess_de(text):
	return [token.text for token in de_nlp.tokenizer(text)]


def preprocess_en(text):
	return [token.text for token in en_nlp.tokenizer(text)]


# Load vocabularies (you need to load or create these separately)
de_vocab = torch.load('saved_paths/de_vocab.pt')
en_vocab = torch.load('saved_paths/en_vocab.pt')

# Define model parameters
input_dim = len(de_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize attention mechanism and model
attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = Encoder(
	input_dim,
	encoder_embedding_dim,
	encoder_hidden_dim,
	decoder_hidden_dim,
	encoder_dropout,
	)

decoder = Decoder(
	output_dim,
	decoder_embedding_dim,
	encoder_hidden_dim,
	decoder_hidden_dim,
	decoder_dropout,
	attention,
	)

# Initialize Seq2Seq model
model = Seq2Seq(encoder, decoder, en_vocab["<pad>"], teacher_forcing_ratio=0.5).to(device)

# Load pretrained model weights (if available)
model.load_state_dict(torch.load('saved_paths/seq2seq_model.pth', map_location=device))

# Set the model to evaluation mode
model.eval()


# Example of how to use the model for translation (adjust according to your implementation)
def translate_sentence(
		sentence,
		model,
		en_nlp,
		de_nlp,
		en_vocab,
		de_vocab,
		lower,
		sos_token,
		eos_token,
		device,
		max_output_length=25,
		):
	model.eval()
	with torch.no_grad():
		if isinstance(sentence, str):
			de_tokens = [token.text for token in de_nlp.tokenizer(sentence)]
		else:
			de_tokens = [token for token in sentence]
		if lower:
			de_tokens = [token.lower() for token in de_tokens]
		de_tokens = [sos_token] + de_tokens + [eos_token]
		ids = de_vocab.lookup_indices(de_tokens)
		tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
		encoder_outputs, hidden = model.encoder(tensor)
		inputs = en_vocab.lookup_indices([sos_token])
		attentions = torch.zeros(max_output_length, 1, len(ids))
		for i in range(max_output_length):
			inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
			output, hidden, attention = model.decoder(
				inputs_tensor, hidden, encoder_outputs
				)
			attentions[i] = attention
			predicted_token = output.argmax(-1).item()
			inputs.append(predicted_token)
			if predicted_token == en_vocab[eos_token]:
				break
		en_tokens = en_vocab.lookup_tokens(inputs)
	return en_tokens, de_tokens, attentions[: len(en_tokens) - 1]


# Title and description
st.title('German to English Translator')
st.write('Enter a German sentence to translate it into English.')

# Input text box
input_text = st.text_area('Input German Text')

# Translate button
if st.button('Translate'):
    if input_text:
        # Perform translation
        translated_tokens, _, _ = translate_sentence(
            input_text,
            model,
            en_nlp,
            de_nlp,
            en_vocab,
            de_vocab,
            lower=True,
            sos_token="<sos>",
            eos_token="<eos>",
            device=device,
            max_output_length=25,
        )
        translated_text = " ".join(translated_tokens).strip("<sos> <eos>")
        st.write('Translated English Text:')
        st.write(translated_text)
    else:
        st.warning('Please enter some text to translate.')

#%%
