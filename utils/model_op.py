

def store_model(model, path):
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path + ".h5")
    print("[store_model] model with {} mu_num stored to {}."
          .format(model.mu_num, path))