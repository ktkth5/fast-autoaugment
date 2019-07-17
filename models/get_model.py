from models.wideresnet import wideresnet40_2, wideresnet28_10

model_dict = {
    "wideresnet40_2": wideresnet40_2,
    "wideresnet28_10": wideresnet28_10
}

def get_model(name):
    return model_dict[name]()

