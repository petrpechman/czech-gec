import os
import argparse

EPOCHS = 2
STEPS_PER_EPOCH = 8


def try_batch_size(batch_size, max_length, epochs, steps_per_epoch, config, text_file) -> bool:
    import batch_pipeline
    batch_pipeline.main(batch_size, max_length, epochs, steps_per_epoch, config, text_file)


def log_data(filename: str, text: str):
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    with open(filename, append_write, encoding="utf-8") as log_file:
        print(text, file=log_file)


def get_batch_size(batch_size: int, max_length, epochs, steps_per_epoch, filename, config, text_file) -> int:
    try:
        try_batch_size(batch_size, max_length, epochs, steps_per_epoch, config, text_file)
        log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
        print(f"Allowed batch size {batch_size} for max_length {max_length}.")
    except Exception as e:
        return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()

    
    # filename = "mt5-small-node1.txt"
    filename = args.output
    
    # config = "../../mt5-small/config-small.json"
    config = args.config

    # Some text file enough length with enough sentence length for try
    # text_file = "./text.txt"
    text_file = args.text

    get_batch_size(args.batch_size, args.max_length, EPOCHS, STEPS_PER_EPOCH, filename, config, text_file)


if __name__ == "__main__":
    main()
