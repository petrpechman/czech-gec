import os

MAX_BATCH_SIZE = 2049
STEP_BATCH = 8
    
def try_batch_size(batch_size, max_length, config, text_file) -> bool:
    import batch_pipeline
    batch_pipeline.main(batch_size, max_length, config, text_file)

def get_batch_size(start: int, max_length, filename, config, text_file) -> int:
    for batch_size in range(start, MAX_BATCH_SIZE, STEP_BATCH):
        try:
            try_batch_size(batch_size, max_length, config, text_file)
            log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
            print(f"Allowed batch size {batch_size} for max_length {max_length}.")
        except:
            return batch_size - STEP_BATCH

def log_data(filename: str, text: str):
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    with open(filename, append_write, encoding="utf-8") as log_file:
        print(text, file=log_file)

def main():
    filename = "mt5-small-batches.txt"
    config = "../mt5/config.json"
    text_file = "./text.txt"

    # Do not use 0 because than the batch size is 0 and it returns error.
    get_batch_size(32, 64, filename)
    get_batch_size(16, 96, filename)
    get_batch_size(16, 128, filename)

if __name__ == "__main__":
    main()
