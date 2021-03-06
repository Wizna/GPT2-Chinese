import torch
import torch.nn.functional as F
import os
import argparse
import string
from tqdm import trange
from transformers import GPT2LMHeadModel
from itertools import combinations


def is_word(word):
    for item in list(word):
        if item not in string.ascii_lowercase:
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    :param logits: logits distribution shape (vocabulary size)
    :param top_k: keep only top k tokens with highest probability (top-k filtering)
    :param top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    :param filter_value: filtered logits
    :return:
     From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence_of_length(model, n, generated, repetition_penalty, top_k, top_p, temperature,
                              tokenizer, repeat_map, starting_point,
                              device,
                              length_of_context,
                              generated_token_prefix_tree):
    traversed_tree = {}
    longer_candidate = None
    tried_num = 0
    while tried_num < 1000 or longer_candidate is None:
        tried_num += 1
        tmp_generated = generated.clone().detach()
        tmp_tree = traversed_tree
        prefix_tree = generated_token_prefix_tree
        same_depth = 0
        index = 0
        while index < n + 1:
            if (index + starting_point) in repeat_map:
                start, span = repeat_map[index + starting_point]
                start += length_of_context
                tmp_generated = torch.cat((tmp_generated, tmp_generated[0][start:start + span].unsqueeze(0)), dim=1)
                index += span
                if index >= n:
                    tmp_generated = torch.cat((tmp_generated, torch.tensor([[102]], device=device)), dim=1)
                    print(f'return directly after repeat')
                    return tmp_generated, index + starting_point + 1
                else:
                    # print(f'keep for next token')
                    continue
            inputs = {'input_ids': tmp_generated[0].unsqueeze(0)}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            # note: logits in log space, range from -inf to around 20.0
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token_logits[list(tmp_tree.keys())] /= (repetition_penalty ** (index + 1))
            next_token_logits[list(prefix_tree.keys())] /= (repetition_penalty ** (same_depth + 1))
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if index == n:
                if filtered_logits[102] != -float('Inf'):
                    print('directly get a sep because it is in the range')
                    next_token = torch.tensor(data=[102], dtype=torch.long, device=device)

            tmp_generated = torch.cat((tmp_generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() not in tmp_tree:
                tmp_tree[next_token.item()] = {}

            if next_token.item() not in prefix_tree:
                prefix_tree = generated_token_prefix_tree
                same_depth = 0

            if next_token.item() in prefix_tree:
                prefix_tree = prefix_tree[next_token.item()]
                same_depth += 1

            tmp_tree = tmp_tree[next_token.item()]

            index += 1
            # note: this is to remove sequence containing improper space or other characters
            if next_token < 143:
                if index == n + 1:
                    tmp_generated[0][-1] = 102
                    return tmp_generated, index + starting_point
                else:
                    print(f'generated a token to sep at: {index - 1}')
                    break
        else:
            longer_candidate = tmp_generated.clone().detach()
            longer_candidate[0][-1] = 102
            print(f'stored current longest candidates')
    else:
        print('use the saved longer candidate')
        return longer_candidate, n + starting_point + 1


def distance_from_next_sep(lyric, start):
    for i in range(start, len(lyric)):
        if lyric[i] == 102:
            return i - start

    return len(lyric) - start


def penalize_sequence_in_prefix_tree(sequence, generated_token_prefix_tree, tokenizer=None):
    # note: only penalize sequence of length >= 3
    # also the sequence contain [102] at the end
    if tokenizer:
        text = tokenizer.convert_ids_to_tokens(sequence)
        text = ''.join(text)
        print(f'to be penalize: {text}')
    for i in range(len(sequence) - 3):
        prefix_tree = generated_token_prefix_tree
        for token in sequence[i:]:
            if token not in prefix_tree:
                prefix_tree[token] = {}

            prefix_tree = prefix_tree[token]


def sample_sequence(model, context, length, tokenizer,
                    temperature=1.0,
                    top_k=30,
                    top_p=0.0,
                    repetition_penalty=1.0,
                    device='cpu',
                    repeat_map=None,
                    model_lyric=None):
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    generated = context

    print(f'the repeat_map: {repeat_map}')
    index = 2

    length_of_context = context.size()[1] - 2  # note: this 2 is compensate for the starting index 2
    print(f'length of previous context: {length_of_context}')

    generated_token_prefix_tree = {}
    to_be_penalize_list = []
    s = index
    for i in range(index, context.size()[1]):
        if context[0][i] == 102:
            to_be_penalize_list.append(context[0][s:i + 1])
            s = i + 1

    for seq in to_be_penalize_list:
        seq_ints = seq.tolist()
        penalize_sequence_in_prefix_tree(sequence=seq_ints,
                                         generated_token_prefix_tree=generated_token_prefix_tree,
                                         tokenizer=None)

    with torch.no_grad():
        while index < length:
            distance = distance_from_next_sep(model_lyric, start=index)
            print(f'generating index: {index}, needed length: {distance}')

            generated, new_index = sample_sequence_of_length(model=model, n=distance, generated=generated,
                                                             repetition_penalty=repetition_penalty,
                                                             top_k=top_k, top_p=top_p, temperature=temperature,
                                                             tokenizer=tokenizer, repeat_map=repeat_map,
                                                             starting_point=index,
                                                             device=device,
                                                             length_of_context=length_of_context,
                                                             generated_token_prefix_tree=generated_token_prefix_tree)
            index = new_index

            to_be_penalize = generated.tolist()[0][-distance - 1:]
            penalize_sequence_in_prefix_tree(sequence=to_be_penalize,
                                             generated_token_prefix_tree=generated_token_prefix_tree,
                                             tokenizer=tokenizer)

    return generated.tolist()[0]


# def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
#     inputs = torch.LongTensor(context).view(1, -1).to(device)
#     if len(context) > 1:
#         _, past = model(inputs[:, :-1], None)[:2]
#         prev = inputs[:, -1].view(1, -1)
#     else:
#         past = None
#         prev = inputs
#     generated = [] + context
#     with torch.no_grad():
#         for _ in trange(length):
#             output = model(prev, past=past)
#             output, past = output[:2]
#             output = output[-1].squeeze(0) / temperature
#             filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
#             next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
#             generated.append(next_token.item())
#             prev = next_token.view(1, 1)
#             if next_token in [101, 102]:
#                 break
#     return generated


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
             device='cpu',
             is_fast_pattern=False,
             repeat_map=None,
             model_lyric=None):
    if is_fast_pattern:
        return None
        # fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
        #                         device=device)
    else:
        return sample_sequence(model=model,
                               context=context,
                               length=length,
                               tokenizer=tokenizer,
                               temperature=temperature,
                               top_k=top_k,
                               top_p=top_p,
                               repetition_penalty=repetition_penalty,
                               device=device,
                               repeat_map=repeat_map,
                               model_lyric=model_lyric)


def find_repeating_sequence_from_list(arr):
    result = {}
    longest = len(arr) // 2
    remain = set(i for i in range(len(arr)) if arr[i] != 102)

    for length in range(longest, 2, -1):
        pos = [p for p in remain if all(i in remain for i in range(p, p + length))]
        for s1, s2 in combinations(pos, 2):
            if s2 - s1 >= length and arr[s1:s1 + length] == arr[s2:s2 + length]:
                remain -= set(range(s1, s1 + length))
                remain -= set(range(s2, s2 + length))
                result[s2] = (s1, length)

    return result


def read_and_tokenize_text(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    raw_text = ''.join(data)
    raw_text = raw_text.replace('\n', ' [SEP] ')
    print(raw_text)
    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
    context_tokens = [tokenizer.convert_tokens_to_ids('[CLS]'),
                      tokenizer.convert_tokens_to_ids('[MASK]')] + context_tokens
    print(context_tokens)
    return context_tokens


def read_original_lyric_as_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--original_lyric', default='original_lyric.txt', type=str, required=False, help='原版歌词，模仿句式')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='萧炎', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    original_lyric_text = read_original_lyric_as_text(args.original_lyric)

    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
    while True:
        context_tokens = read_and_tokenize_text(args.prefix, tokenizer=tokenizer)
        lyric_tokens = read_and_tokenize_text(args.original_lyric, tokenizer=tokenizer)
        repeat_map = find_repeating_sequence_from_list(lyric_tokens)
        length = len(lyric_tokens)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = generate(n_ctx=n_ctx,
                           model=model,
                           context=context_tokens,
                           length=length,
                           is_fast_pattern=args.fast_pattern,
                           tokenizer=tokenizer,
                           temperature=temperature,
                           top_k=topk,
                           top_p=topp,
                           repetition_penalty=repetition_penalty,
                           device=device,
                           repeat_map=repeat_map,
                           model_lyric=lyric_tokens)
            # for i in range(batch_size):
            generated += 1
            text = tokenizer.convert_ids_to_tokens(out)
            text = text[len(context_tokens):]
            for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                if is_word(item) and is_word(text[i + 1]):
                    text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'
            info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
            print(info)
            text = ''.join(text).replace('##', '').strip()

            original_lyric = 'original:'
            generated_lyric = 'generated:'
            print(f'{original_lyric:40}', end='')
            print(f'{generated_lyric}')
            generated_lyric_lines = text.split('\n')

            for i, j in zip(original_lyric_text, generated_lyric_lines):
                to_length = 40 - len(i.replace(' ', ''))
                print(i.ljust(to_length), end='')
                print(f'{j}')

            print('')
            if args.save_samples:
                samples_file.write(info)
                samples_file.write(text)
                samples_file.write('\n')
                samples_file.write('=' * 90)
                samples_file.write('\n' * 2)
        print("=" * 80)
        if generated == nsamples:
            # close file when finish writing.
            if args.save_samples:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
