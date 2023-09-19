import os

MAX_BATCH_SIZE = 2049
STEP_BATCH = 4
    
def try_batch_size(batch_size, max_length, epochs, steps_per_epoch, config, text_file) -> bool:
    import batch_pipeline
    batch_pipeline.main(batch_size, max_length, epochs, steps_per_epoch, config, text_file)

def get_batch_size(start: int, max_length, epochs, steps_per_epoch, filename, config, text_file) -> int:
    for batch_size in range(start, MAX_BATCH_SIZE, STEP_BATCH):
        try:
            try_batch_size(batch_size, max_length, epochs, steps_per_epoch, config, text_file)
            log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
            print(f"Allowed batch size {batch_size} for max_length {max_length}.")
        except Exception as e:
            print(e)
            return batch_size - STEP_BATCH

def log_data(filename: str, text: str):
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    with open(filename, append_write, encoding="utf-8") as log_file:
        print(text, file=log_file)

def main():
    filename = "mt5-small-node1.txt"
    config = "../mt5-small/config-small.json"
    text_file = "./text.txt"
    epochs = 2
    steps_per_epoch = 4

    # Do not use 0 because than the batch size is 0 and it returns error.
    max_lenghts = [32, 64, 96, 128]
    for max_length in max_lenghts:
        batch_size = get_batch_size(4, max_length, epochs, steps_per_epoch, filename, config, text_file)
        print(f"For max_lenght {max_length} is batch_size {batch_size}.")

if __name__ == "__main__":
    main()
