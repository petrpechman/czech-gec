import argparse
import pipeline
import evaluator
import generate_error_data
import create_dataset

import pipeline_print_output

import evaluator_pred
import evaluator_files

def main(config_filename: str, eval: bool, generate: bool = False, create: bool = False, print_flag: bool = False,
         part1: bool = False, part2: bool = False):
    if (eval and generate) or (eval and create) or (create and generate):
        print("It is not possible to use eval, generate or create together...")

    if generate:
        print("USE GENERATE...")
        generate_error_data.main(config_filename)
    elif create:
        print("USE CREATE...")
        create_dataset.main(config_filename)
    elif eval:
        if part1:
            evaluator_pred.main(config_filename)
        elif part2:
            evaluator_files.main(config_filename)
        else:
            evaluator.main(config_filename)
    else:
        if print_flag:
            pipeline_print_output.main(config_filename)
        else:
            pipeline.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run evaluation.")
    parser.add_argument("--part1", action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--part2", action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, help="Generate by reverted pipeline.")
    parser.add_argument("--create", action=argparse.BooleanOptionalAction, help="Create dataset.")
    parser.add_argument("--print", action=argparse.BooleanOptionalAction, help="Print dataset.")
    args = parser.parse_args()

    print(args.eval)
    main(args.config, args.eval, args.generate, args.create, args.print, args.part1, args.part2)