import json

def get_config(path):
    with open(path, 'r') as file:
        config = json.load(file)

    config["data_path"] = f'data/data/{config["data"]}.json'
    config["data_info_path"] = f'data/data/{config["data"]}_info.json'
    config["data_weight_path"] = f'data/data/{config["data"]}_weight.pkl'
    config["tensor_board_folder"] = f'runs/{config["data"]}/{config["name"]}'
    # config["save_folder"] = f'./saves/{config["data"]}/{config["name"]}'
    config["save_folder"] = f'saves/{config["data"]}/{config["name"]}'
    config["evaluate_path"] = f'evaluate/{config["data"]}/{config["name"]}.json'
    config["predict_path"] = f'predict/{config["data"]}/{config["name"]}.csv'

    with open(config["data_info_path"], 'r') as file:
        data_info_dict = json.load(file)

    config["items_total"] = data_info_dict['num_items']
    config["users_total"] = data_info_dict['num_users']
    if "items_total" in config["model"]: config["model"]["items_total"] = data_info_dict["num_items"]
    if "users_total" in config["model"]: config["model"]["users_total"] = data_info_dict['num_users']

    return config