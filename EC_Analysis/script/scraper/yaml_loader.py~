import yaml
import sys


def load_yaml(filename):

    with open(filename, encoding='utf_8_sig') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    return config


def main():

    config = load_yaml(sys.argv[1])

    print(config)


if __name__ == '__main__':
    main()
