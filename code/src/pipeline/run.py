import argparse
import pipeline
import evaluator
import generate_error_data
import create_dataset

import old_pipeline
import old_evaluator

def main(config_filename: str, eval: bool, generate: bool = False, create: bool = False, old: bool = False):
    if (eval and generate) or (eval and create) or (create and generate):
        print("It is not possible to use eval, generate or create together...")

    if generate:
        print("USE GENERATE...")
        generate_error_data.main(config_filename)
    elif create:
        print("USE CREATE...")
        create_dataset.main(config_filename)
    elif eval:
        if args.old:
            old_evaluator.main(config_filename)
        else:
            evaluator.main(config_filename)
    else:
        if args.old:
            old_pipeline.main(config_filename)
        else:
            pipeline.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run evaluation.")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, help="Generate by reverted pipeline.")
    parser.add_argument("--create", action=argparse.BooleanOptionalAction, help="Create dataset.")
    parser.add_argument("--old", action=argparse.BooleanOptionalAction, help="Use old code...")
    args = parser.parse_args()

    print(args.eval)
    main(args.config, args.eval, args.generate, args.create, args.old)