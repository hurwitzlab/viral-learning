from keras.models import model_from_json

from vl.model.training import train_and_evaluate, build_model
from vl.data.kmers import BacteriaAndVirusKMers


def main():
    network_depths = (
        (('Dense', {'units': 128, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 64, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 32, 'activation': 'relu'}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
    )

    the_data = BacteriaAndVirusKMers(
        fp='/home/jklynch/host/project/viral-learning/data/perm_training_testing.h5',
        training_sample_count=1000,
        development_sample_count=2000,
        half_batch_size=100)

    model_name, model = build_model(input_dim=the_data.get_input_dim(), layers=network_depths[0])

    training_metrics_df, dev_metrics_df = train_and_evaluate(
        model=model,
        model_name=model_name,
        training_epochs=10,
        the_data=the_data
    )

    # store the model
    with open(model_name + '.json', 'wt') as model_json:
        model_json.write(model.to_json())

    model.save_weights(filepath=model_name + '.h5', overwrite=True)

    # sanity check
    with open(model_name + '.json', 'rt') as model_json:
        saved_model = model_from_json(model_json.read())

    saved_model.load_weights(filepath=model_name + '.h5')


if __name__ == '__main__':
    main()
