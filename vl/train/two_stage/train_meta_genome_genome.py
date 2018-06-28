from pprint import pprint

from keras import Sequential

from vl.model.training import train_and_evaluate, build_model
from vl.data.kmers import BacteriaAndVirusKMers
from vl.data.kmers_genome import BacteriaAndVirusGenomeKMers


def main():
    first_stage_network_depths = (
        (('Dense', {'units': 128, 'activation': 'relu'}),
         ('Dropout', {'rate': 0.4}),
         ('Dense', {'units': 64, 'activation': 'relu'}),
         ('Dense', {'units': 1, 'activation': 'sigmoid'})),
    )

    first_stage_data = BacteriaAndVirusKMers(
        fp='/home/jklynch/host/project/viral-learning/data/perm_training_testing.h5',
        training_sample_count=100000,
        development_sample_count=1000,
        half_batch_size=50)

    first_stage_model_name, first_stage_model = build_model(
        model=Sequential(),
        input_dim=first_stage_data.get_input_dim(),
        layers=first_stage_network_depths[0])

    first_stage_model_name = 'first_stage_' + first_stage_model_name

    training_metrics_df, dev_metrics_df = train_and_evaluate(
        model=first_stage_model,
        model_name=first_stage_model_name,
        training_epochs=5,
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
    second_stage_layers = (
        (
            #('Dense', {'units': 64, 'activation': 'relu'}),
            ('Dense', {'units': 1, 'activation': 'sigmoid'}), ),
    )
    second_stage_model_name, second_stage_model = build_model(
        model=second_stage_model,
        layers=second_stage_layers[0])

    second_stage_model_name = 'second_stage_' + second_stage_model_name

    second_stage_data = BacteriaAndVirusGenomeKMers(
        fp='/home/jklynch/host/project/viral-learning/data/riveal_refseq_prok_phage_500pb_kmers8.h5',
        pb=500,
        k=8,
        training_sample_count=100000,
        development_sample_count=1000,
        half_batch_size=50)

    pprint(second_stage_model.get_config())

    genomic_training_metrics_df, genomic_dev_metrics_df = train_and_evaluate(
        model=second_stage_model,
        model_name=second_stage_model_name,
        training_epochs=5,
        the_data=second_stage_data
    )

    # store the model
    with open(second_stage_model_name + '.json', 'wt') as model_json:
        model_json.write(second_stage_model.to_json())
    second_stage_model.save_weights(filepath=second_stage_model_name + '.h5', overwrite=True)


if __name__ == '__main__':
    main()
