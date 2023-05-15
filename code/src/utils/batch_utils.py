import os
    
def try_batch_size(batch_size, max_length, num_lines, lr=0.00001) -> bool:
    import batch_pipeline_mt5
    batch_pipeline_mt5.main(batch_size, max_length, "../transformer/config.json", num_lines)

def get_batch_size(max_length, filename) -> int:
    NUM_LINES = 128
    MAX_BATCH_SIZE = 2049
    STEP_BATCH = 4

    for batch_size in range(STEP_BATCH, MAX_BATCH_SIZE, STEP_BATCH):
        try:
            try_batch_size(batch_size, max_length, NUM_LINES)
            log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
            print(f"Allowed batch size {batch_size} for max_length {max_length}.")
        except:
            return batch_size - STEP_BATCH
    return 0
        
def all_batch_sizes(filename: str):
    MAX_LENGTH = 16384
    STEP_LENGTH = 16

    batch_sizes = []
    for max_length in range(STEP_LENGTH, MAX_LENGTH, STEP_LENGTH):
        batch_size = get_batch_size(max_length, filename)
        print(f"BATCH SIZE: {batch_size}   MAX LENGHT: {max_length}")
        if batch_size == 0:
            break
        batch_sizes.append((max_length, batch_size))
    return batch_sizes

def log_data(filename: str, text: str):
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w'
    with open(filename, append_write, encoding="utf-8") as log_file:
        print(text, file=log_file)

def main():
    filename = "mt5-small-batches-2.txt"
    get_batch_size(32 ,filename)
    get_batch_size(64 ,filename)
    get_batch_size(96 ,filename)
    get_batch_size(128 ,filename)

if __name__ == "__main__":
    main()
