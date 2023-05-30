import argparse
import pipeline
import evaluator

def main(config_filename: str, eval: bool):
    if eval == True:
        evaluator.main(config_filename)
    else:
        pipeline.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=str, help="Config file.")
    parser.add_argument("eval", type=bool, help="Run evaluation.")
    args = parser.parse_args()
    main(args.config, args.eval)