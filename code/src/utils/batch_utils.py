import os
    
def try_batch_size(batch_size, max_length, lr=0.00001) -> bool:
    import batch_pipeline
    batch_pipeline.main(batch_size, max_length, "../transformer/config.json", "./text.txt")

def get_batch_size(start: int, max_length, filename) -> int:
    MAX_BATCH_SIZE = 2049
    STEP_BATCH = 8
    
    for batch_size in range(start, MAX_BATCH_SIZE, STEP_BATCH):
        try:
            try_batch_size(batch_size, max_length)
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
    # !! not use 0
    get_batch_size(88, 32 ,filename)
    get_batch_size(28, 64 ,filename)
    get_batch_size(16, 96 ,filename)
    get_batch_size(12, 128 ,filename)

if __name__ == "__main__":
    main()
