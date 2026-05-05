import sys

from t_dev_810.experiments import build_config, runner


def main():
    try:
        experimentConf = build_config()
        runner(experimentConf)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
