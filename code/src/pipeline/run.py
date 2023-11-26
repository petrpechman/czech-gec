import argparse
import pipeline
import evaluator
import generate_error_data

def main(config_filename: str, eval: bool, generate: bool = False):
    if eval and generate:
        print("It is not possible to use eval and generate together...")
    if generate:
        print("USE GENERATE...")
        generate_error_data.main(config_filename)
    elif eval:
        evaluator.main(config_filename)
    else:
        pipeline.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run evaluation.")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, help="Generate by reverted pipeline.")
    args = parser.parse_args()

    print(args.eval)
    main(args.config, args.eval, args.generate)