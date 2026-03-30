import sys

from t_dev_810.experiments import build_config
from t_dev_810.experiments.runner import runner


def main():
    try:
        experimentConf = build_config()
        runner(experimentConf)
    except Exception as e:
        print(e)
        sys.exit


if __name__ == "__main__":
    main()
