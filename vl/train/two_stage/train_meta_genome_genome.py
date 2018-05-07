from pprint import pprint

from keras import Sequential

from vl.model.training import train_and_evaluate, build_model
from vl.data.kmers import BacteriaAndVirusKMers


def main():
    first_stage_network_depths = (
        (('Dense', {'units': 128, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 64, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.2}),
         ('Dense', {'units': 32, 'activation': 'relu'}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
    )

    first_stage_data = BacteriaAndVirusKMers(
        fp='/home/jklynch/host/project/viral-learning/data/perm_training_testing.h5',
        training_sample_count=1000,
        development_sample_count=2000,
        half_batch_size=100)

    first_stage_model_name, first_stage_model = build_model(
        model=Sequential(),
        input_dim=first_stage_data.get_input_dim(),
        layers=first_stage_network_depths[0])

    training_metrics_df, dev_metrics_df = train_and_evaluate(
        model=first_stage_model,
        model_name=first_stage_model_name,
        training_epochs=10,
        the_data=first_stage_data
    )

    pprint(first_stage_model.get_config())

    # store the model
    with open(first_stage_model_name + '.json', 'wt') as model_json:
        model_json.write(first_stage_model.to_json())

    first_stage_model.save_weights(filepath=first_stage_model_name + '.h5', overwrite=True)

    second_stage_model = Sequential()
    second_stage_model.add(first_stage_model.get_layer(index=0))
    second_stage_model.add(first_stage_model.get_layer(index=1))
    second_stage_model.add(first_stage_model.get_layer(index=2))
    second_stage_model.add(first_stage_model.get_layer(index=3))
    second_stage_layers = (
        (('Dense', {'units': 32, 'activation': 'relu'}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
    )
    second_stage_model_name, second_stage_model = build_model(
        model=second_stage_model,
        layers=second_stage_layers[0])

    second_stage_data = BacteriaVirusGenomeKmers(
        fp='/home/jklynch/host/project/viral-learning/data/perm_training_testing.h5',
        training_sample_count=1000,
        development_sample_count=2000,
        half_batch_size=100)

    pprint(second_stage_model.get_config())


if __name__ == '__main__':
    main()
