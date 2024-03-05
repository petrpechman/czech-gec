import argparse
import pipeline
import evaluator
import generate_error_data
import create_dataset

import old_pipeline
import old_evaluator

import pipeline_v1
import pipeline_v2
import pipeline_v3

def main(config_filename: str, eval: bool, generate: bool = False, create: bool = False, old: bool = False,
         version: int = 0):
    if (eval and generate) or (eval and create) or (create and generate):
        print("It is not possible to use eval, generate or create together...")

    if generate:
        print("USE GENERATE...")
        generate_error_data.main(config_filename)
    elif create:
        print("USE CREATE...")
        create_dataset.main(config_filename)
    elif eval:
        if old:
            old_evaluator.main(config_filename)
        else:
            evaluator.main(config_filename)
    else:
        if old:
            old_pipeline.main(config_filename)
        else:
            if version == 0:
                pipeline.main(config_filename)
            elif version == 1:
                pipeline_v1.main(config_filename)
            elif version == 2:
                pipeline_v2.main(config_filename)
            elif version == 3:
                pipeline_v3.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run evaluation.")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, help="Generate by reverted pipeline.")
    parser.add_argument("--create", action=argparse.BooleanOptionalAction, help="Create dataset.")
    parser.add_argument("--old", action=argparse.BooleanOptionalAction, help="Use old code...")
    parser.add_argument("--version", type=int, help="Pipeline version.", default=0)
    args = parser.parse_args()

    print(args.eval)
    main(args.config, args.eval, args.generate, args.create, args.old, args.version)