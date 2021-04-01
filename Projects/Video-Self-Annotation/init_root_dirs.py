import os
import ulti
import argparse


def init_data(experiment_type='Entire_dataset', iter=0):
        root_dir = os.path.join(os.getcwd(), 'CityScapes/val')
        dataset_dir = os.path.join(root_dir, 'Dataset')
        training_dir = os.path.join(root_dir, 'Train')
        dataset_name = 'CityScapes_val'
        annotated_video = dataset_name

        experiment = os.path.join(experiment_type, 'Iter_{}'.format(iter))

        ########################################################

        info = {'dataset_dir': dataset_dir,
                'dataset_name': dataset_name,
                'training_dir': training_dir,
                'annotated_video': annotated_video,
                'experiment': experiment,
                'iter': iter,
                }

        ulti.make_dir(os.path.join(dataset_dir, 'Images'))
        ulti.make_dir(training_dir)

        outfile = ulti.write_json(info)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_type", type=str, default='Entire_dataset')
        parser.add_argument("--iter", type=int, default=0)
        args = parser.parse_args()
        init_data(args.experiment_type, args.iter)

