import argparse
from retag import main

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your CLI tool")
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default='out.m2')
    parser.add_argument('-c', '--count', action='store_true')
    return parser.parse_args()

def main_cli():
    args = parse_args()
    main(args)

if __name__ == "__main__":
    main_cli()
