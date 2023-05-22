import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as file:
        content = file.read()

    items = content.split('\n\n')
    items[-1] = items[-1].strip('\n')

    sorted_items = sorted(items, key=lambda item: len(item.split('\n')[0]), reverse=True)

    sorted_content = '\n\n'.join(sorted_items)

    with open(args.output, 'w') as file:
        file.write(sorted_content)
        file.write("\n")

if __name__ == "__main__":
    main()

#CMD: python3 sort_m2.py --input "../../data/geccc/dev/sentence.m2" --output "../../data/geccc/dev/sorted_sentence.m2"