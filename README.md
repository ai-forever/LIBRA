In order to add your own set, you must first add it to the configs/datasets_config.json file.
Then create a config for it like longchat32k.ini and specify the necessary parameters in it.

To run the script to generate answers to the tasks, you need to run the following command:
python main.py -c path_to_config

For metric evaluation you need to run next command:
python eval.py -p path_to_predictions