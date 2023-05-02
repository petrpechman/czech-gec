import argparse
from pathlib import Path
from typing import Any

from tokenizers import BertWordPieceTokenizer
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--min-frequency", type=int, default=5)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    tokenizer.train(
        args.corpus_file,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        # Fairseq has fixed order of special tokens: bos, pad, eos, unk (mask
        # is handled separately)
        # The order here can be different, because we reorder them when
        # creating the final vocab.txt (in Makefile)
        special_tokens=["<pad>", "</s>", "<cls>", "<unk>", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    # it would be better to also save tokenizer config etc. with lowercase=True
    # and strip_accents=False, but this version doesn't seem to support
    # save_pretrained()
    tokenizer.save_model(args.output_dir)
    tokenizer.save(args.output_dir + "tokenizer.json")

class CustomTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos = tokenizer.token_to_id("</s>")
        self.bos = tokenizer.token_to_id("<cls>")

    def __call__(self, text, text_target, max_length, **kwargs):
        encoded_text = self.tokenizer.encode(text)

        input_ids = encoded_text.ids[:(max_length-2)]
        input_ids.insert(0, self.bos)
        input_ids.append(self.eos)

        attention_mask = encoded_text.attention_mask[:(max_length-2)]
        attention_mask.insert(0, 1)
        attention_mask.append(1)


        encoded_text_target = self.tokenizer.encode(text_target)

        labels = encoded_text_target.ids[:(max_length-2)]
        labels.insert(0, self.bos)
        labels.append(self.eos)

        input_ids = tf.convert_to_tensor(input_ids, tf.int32)
        attention_mask = tf.convert_to_tensor(attention_mask, tf.int32)
        labels = tf.convert_to_tensor(labels, tf.int32)

        output = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "labels": [labels]
        }

        return output


if __name__ == "__main__":
    main()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("./out/tokenizer.json")
    output = tokenizer.encode("Podobal se srdci", add_special_tokens=True)
    print(output.ids)
    print(output.attention_mask)
    print(output.special_tokens_mask)
    print(tokenizer.token_to_id("[CLS]"))

    my_tokenizer = CustomTokenizer(tokenizer)
    output = my_tokenizer("Podobal se srdci", "Podobal s srdci", 10)
    print(output)


