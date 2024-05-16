import os
import warnings
import logging
logging.disable(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.resetwarnings()
warnings.simplefilter('ignore')
import keras
import gensim
import re


def prepare_input_text(input_text):
    ret = input_text.lower()
    ret = re.sub("\'", " ' ", ret)
    ret = re.sub("\?", " ? ", ret)
    ret = re.sub("!", " ! ", ret)
    ret = re.sub("\.", " . ", ret)
    ret = re.sub(",", " , ", ret)
    ret = re.sub(";", " ; ", ret)
    ret = re.sub(":", " : ", ret)
    return ret


def correct_token_vector_length(token_vector, input_length):
    if len(token_vector) >= input_length:
        return token_vector[-input_length:]
    ret = ["***"] * (input_length - len(token_vector)) + token_vector
    return ret


def check_input_language(tokens, key_to_index):
    non_english_counter = 0
    for t in tokens:
        if t not in key_to_index:
            non_english_counter += 1
            if non_english_counter > int(len(tokens) / 6):
                return False
    return True


def make_embedding_input(text, key_to_index):
    ret = [key_to_index[t if t in key_to_index else "***"] for t in text]
    return ret


def make_predictions(prediction_input, max_prediction_length, w2v_model, keras_model):
    ret = []
    embedding_input = prediction_input.copy()
    for _ in range(max_prediction_length):
        prediction_vector = keras_model.predict([embedding_input], verbose=False)
        prediction_word = w2v_model.wv.most_similar(prediction_vector)[0][0]
        embedding_input.append(key_to_index[prediction_word])
        embedding_input.pop(0)
        if len(ret) == 0 or not prediction_word == ret[-1]:
            ret.append(prediction_word)
        if prediction_word in ('.', '!', '?'):
            break
    return ret


def removeAsterixs(tokens):
    for i, t in tokens:
        if not t == "***":
            break
    ret = tokens[i:]
    return ret


def delete_repetitions(tokens):
    ret = tokens.copy()
    cut = True
    while cut:
        cut = False
        for start in range(0, len(ret) - 1):
            for end in range(start + 1, len(ret)):
                if end - start > len(ret) - end:
                    break
                pattern = ret[start:end]
                neighbour = ret[end:end + (end - start)]
                if pattern == neighbour:
                    ret = ret[0:end] + ret[end + (end - start):len(ret)]
                    cut = True
                    break
            if cut:
                break
    return ret


def tokens_to_text(tokens):
    ret = " ".join(tokens)
    ret = re.sub(" ' ", "' ", ret)
    ret = re.sub(" \?", "?", ret)
    ret = re.sub(" !", "!", ret)
    ret = re.sub(" \.", ".", ret)
    ret = re.sub(" , ", ", ", ret)
    ret = re.sub(" ; ", "; ", ret)
    ret = re.sub(" : ", ": ", ret)
    ret = re.sub(" s ", "s ", ret)
    return ret


def reply_to_input_text(input_vector, input_text, w2v_model, keras_model):
    prepared_text = prepare_input_text(input_text)
    new_vector = prepared_text.split()
    if not check_input_language(new_vector, w2v_model.wv.key_to_index):
        return "this is not proper english.", input_vector
    ret_vector = input_vector + new_vector
    if not len(ret_vector) > 90:
        return "i need a longer input, keep on typing!", ret_vector
    ret_vector = correct_token_vector_length(ret_vector, model_input_length)
    embedding_input = make_embedding_input(ret_vector, key_to_index)
    output_tokens = make_predictions(embedding_input, 100, w2v_model, keras_model)
    output_tokens = delete_repetitions(output_tokens)
    if not len(output_tokens) > 5:
        return "i do not know an answer to that.", ret_vector
    ret_text = tokens_to_text(output_tokens)
    ret_vector += output_tokens
    return ret_text, ret_vector


if __name__ == '__main__':
    print("wait! ... i am loading ...")
    keras_model = keras.models.load_model("keras_model.keras")
    model_input_length = keras_model.input_shape[1]
    w2v_model = gensim.models.Word2Vec.load("w2vModel", mmap='r')
    index_to_key = w2v_model.wv.index_to_key
    key_to_index = w2v_model.wv.key_to_index
    print("... finished loading.")
    input_vector = []
    while True:
        print()
        console_input_text = input()
        if console_input_text.strip() in ["exit", "quit"]:
            exit()
        reply, input_vector = reply_to_input_text(input_vector, console_input_text, w2v_model, keras_model)
        print()
        print(reply)





