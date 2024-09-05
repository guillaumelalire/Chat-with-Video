import pytubefix as pt
import whisper
import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def extract_audio(filepath, url, output_filepath):
    """
    Generates a .mp3 file from either the provided file or the provided YouTube link.
    Returns success.
    """
    if filepath and filepath != "": # File
        return True
    elif url and url != "": # YouTube link
        yt = pt.YouTube(url)
        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(filename=output_filepath)
        return True
    else:
        return False

def transcript(filepath):
    """
    Transcript audio file using Whisper.
    Calls `translate_to_english()`.
    Returns transcripted text.
    """
    model = whisper.load_model("base")
    
    result = model.transcribe(filepath)
    text = result['text']
    language = result['language']
    
    return text, language

def split_text_into_chunks(text, chunk_size):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text by sentence-ending punctuation followed by a space
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence exceeds the chunk size, save the current chunk and start a new one
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def translate_to_english(text, language):
    """
    Returns translated text
    """
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = language

    CHAR_LIMIT = 800
    chunks = split_text_into_chunks(text, CHAR_LIMIT)

    translated_text = ""

    for chunk in chunks:
        encoded_hi = tokenizer(chunk, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_hi,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        translated_text += tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0] + " "
    
    return translated_text