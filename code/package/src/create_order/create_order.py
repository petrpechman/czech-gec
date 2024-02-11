import os

from tensorflow.python.summary import summary_iterator


def main():
    dir_path = os.path.join(os.getcwd(), "tmp/checkpoint")
    dirs = os.listdir(dir_path)

    for dir in dirs:
        path = os.path.join(dir_path, dir)
        os.chdir(path)
        files = os.listdir(path)

        steps = dict()

        for file in files:
            for event in summary_iterator.summary_iterator(os.path.join(path, file)):
                step = event.step
                print(file)
                print("Step number:", step)
                steps[file] = step

        for filename, step in steps.items():
            step = str(step)
            if len(step) == 1:
                step = "00" + step
            elif len(step) == 2:
                step = "0" + step

            if not filename[:3].isnumeric():
                os.rename(filename, str(step) + '.' + filename)


def main_cli():
    main()


if __name__ == "__main__":
    main_cli()