# Classy Evolutionary Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# main module

import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorFlipMutation, \
    BitStringVectorNFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation

# from networks import PretrainedModels
from torch_datasets import Datasets
from generate_outputs import get_args, timm_models
from utils import acc_per_class, rndstr
from classy_pre import ClassyEnsemblePre

PretrainedModels = ['efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_y_400mf', 'regnet_y_8gf', 'resnet152', 'resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'vgg19_bn'] + timm_models
n_models = len(PretrainedModels)
population_size = 200
max_generation = 20
genome_size = n_models


class EnsembleEvaluator(SimpleIndividualEvaluator):
    def __init__(self, models, n_classes, topk, targets, ftype):
        self.models = models
        self.n_classes = n_classes
        self.topk = topk
        self.targets = targets
        self.ftype = ftype

    def _evaluate_individual(self, individual):  # fitness computation
        models = [self.models[i] for i, v in enumerate(individual.vector) if v == 1]
        if len(models) < self.topk:
            return 0

        ens = ClassyEnsemblePre(models=models, n_classes=self.n_classes, topk=self.topk)
        pred = ens.predict(train=True)
        acc = accuracy_score(self.targets, pred)

        if self.ftype == 1:
            fitness = acc
        elif self.ftype == 2:
            fitness = acc + 1 / ens.size()
        elif self.ftype == 3:
            outputs = np.array([ens.ensemble[e]['predictions_train'] for e in ens.ensemble])
            similarity = cosine_similarity(outputs).mean()
            fitness = acc - similarity

        return fitness


def load_from_csv(f):
    data = np.loadtxt(f, delimiter=',')
    outputs, predictions, targets = data[:, :-2], data[:, -2].astype(int), data[:, -1].astype(int)
    return outputs, predictions, targets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', type=str, action='store', default='cifar10',
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', default='Results',
                        help='directory where results are placed (default: Results)')
    parser.add_argument('-ftype', dest='ftype', type=int, action='store', default=1,
                        help='Type of fitness function (default: 1)')
    parser.add_argument('-topk', dest='topk', type=int, action='store', default=3,
                        help='How many top models per class in ensemble (default: 3)')
    args = parser.parse_args()
    return args.dataset, args.resdir, args.ftype, args.topk


def main():
    dataset, resdir, ftype, topk = get_args()
    assert ftype in [1, 2, 3]
    n_classes = Datasets[dataset]['n_classes']
    resfile = f'{resdir}/cleen-{dataset}-{ftype}-{topk}-{rndstr()}.txt'

    with open(resfile, 'w') as f:
        print(f'dataset: {dataset}', file=f)
        print(f'ftype: {ftype}', file=f)
        print(f'topk: {topk}', file=f)
        print(f'n_classes: {n_classes}', file=f)
        print(f'population_size: {population_size}', file=f)
        print(f'max_generation: {max_generation}', file=f)
        print(f'genome_size: {genome_size}', file=f)
        print(f'n_models: {n_models}', file=f)
        print(f'PretrainedModels: {PretrainedModels}', file=f)

    models, targets_train, targets_test = [], None, None
    for i, model_name in enumerate(PretrainedModels):
        outputs_train, predictions_train, targets_train = load_from_csv(f'{dataset}/{model_name}-train.csv')
        outputs_test, predictions_test, targets_test = load_from_csv(f'{dataset}/{model_name}-test.csv')
        models.append({'model': model_name,
                       'model_num': i,
                       'outputs_train': outputs_train,
                       'outputs_test': outputs_test,
                       'predictions_train': predictions_train,
                       'predictions_test': predictions_test,
                       'score': accuracy_score(targets_train, predictions_train),
                       'test_score': accuracy_score(targets_test, predictions_test),
                       'class_scores': acc_per_class(predictions_train, targets_train, n_classes)})

    srt_train = sorted(models, key=lambda d: d['score'], reverse=True)
    srt_test = sorted(models, key=lambda d: d['test_score'], reverse=True)
    with open(resfile, 'a') as f:
        print(f'best train: {srt_train[0]["score"]}, {srt_train[0]["model"]} ', file=f)
        print(f'best test: {srt_test[0]["test_score"]}, {srt_test[0]["model"]} ', file=f)


    # run evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=genome_size),
                      population_size=population_size,
                      evaluator=EnsembleEvaluator(models=models, n_classes=n_classes, topk=topk,
                                                  targets=targets_train, ftype=ftype),
                      higher_is_better=True,
                      elitism_rate=1/population_size,
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.5, k=1),
                          BitStringVectorNFlipMutation(probability=0.2, probability_for_each=0.05, n=genome_size)
                      ],
                      selection_methods=[
                          (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=max_generation,
        statistics=BestAverageWorstStatistics()
    )

    algo.evolve()  # evolve the generated initial population

    # Execute (show) the best solution
    best_of_run = algo.execute()
    models = [models[i] for i, v in enumerate(best_of_run) if v == 1]
    ens = ClassyEnsemblePre(models=models, n_classes=n_classes, topk=topk)
    pred = ens.predict(train=False)
    test_acc = accuracy_score(targets_test, pred)

    with open(resfile, 'a') as f:
        print(f'best_of_run: {[m["model"] for m in models]}', file=f)
        print(f'best_of_run test acc: {test_acc}', file=f)
        print(f'ens.size: {ens.size()}', file=f)


##############
if __name__ == "__main__":
    main()

