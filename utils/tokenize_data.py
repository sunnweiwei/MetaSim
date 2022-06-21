import sys

sys.path += ['./']
from utils.io import read_dialog, write_pkl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import copy


def tokenize(filename):
    from modeling.config import sgd_action_old_to_new, sgd_action_to_code, satisfaction_to_code, satisfaction_old_to_new
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    data = read_dialog(filename)
    toked_data = []
    for session in tqdm(data):
        dialog = []
        prev_action = sgd_action_old_to_new['other']
        for line in session:
            speaker, text, action = line.split('\t')

            speaker = speaker.lower()
            speaker_code = [tokenizer.encode(speaker)[0]] + [6]

            action = action.split(',')
            action_code = [sgd_action_old_to_new[a] for a in action]
            action_code = action_code[0]
            text_code = tokenizer.encode(text)

            if speaker == 'user':
                #           [1139, 6]      [11153, 6, 0] [21820, 296, 1]
                line_code = speaker_code + action_code + text_code

                #           [119, 6, 0]   [1139, 6, 11153, 6, 0, 21820, 296, 1]
                line_code = prev_action + line_code
            else:
                #           [358, 6]       [21820, 296, 1]
                line_code = speaker_code + text_code
                #           [119, 6, 0]   [358, 6, 21820, 296, 1]
                line_code = prev_action + line_code

            dialog.append(line_code)

            if speaker == 'user':
                toked_data.append(copy.deepcopy(dialog))

            prev_action = action_code
    return toked_data


def main():
    i = 'dataset/tod/sgd/dev.txt'
    o = 'dataset/tod/sgd/dev_aat.pkl'
    data = tokenize(i)
    write_pkl(data, o)


if __name__ == '__main__':
    main()
