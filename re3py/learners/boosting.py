from .tree import DecisionTree, create_constant_tree
from .predictive_model import TreeEnsemble
from .core.heuristic import *
from math import exp, log
from ..ranking.ensemble_ranking import EnsembleRanking
from ..data.data_and_statistics import get_all_target_values


class GradientBoostingTask:
    heuristic = HeuristicVariance()

    @staticmethod
    def create_default_model(data: Dataset):
        raise NotImplementedError("This should be implemented by a subclass.")

    @staticmethod
    def minus_partial_derivative(true_values, predictions):
        raise NotImplementedError("This should be implemented by a subclass.")

    @staticmethod
    def modify_tree(shrinkage, step, optimize_step, tree: DecisionTree):
        """
        Boosting node statistics take care of the optimal predictions in the leaves, i.e.,
        the optimal step is 1, thus predictions in the leaves of the tree are multiplied by
        shrinkage if we optimize for step, and shrinkage * step otherwise.
        :param shrinkage:
        :param step:
        :param optimize_step:
        :param tree:
        :return: None
        """
        if optimize_step:
            step = 1.0
        factor = shrinkage * step
        if factor != 1.0:
            for node in tree:
                if node.is_leaf():
                    current_prediction = node.get_stats().get_prediction()
                    node.get_stats().set_prediction(current_prediction *
                                                    factor)

    @staticmethod
    def is_classification():
        return False

    @staticmethod
    def is_binary_classification():
        return False

    @staticmethod
    def is_multiclass_classification():
        return False

    @staticmethod
    def is_regression():
        return False


class GradientBoostingRegression(GradientBoostingTask):
    @staticmethod
    def create_default_model(data: Dataset):
        return create_constant_tree(GradientBoostingTask.heuristic, data)

    @staticmethod
    def minus_partial_derivative(true_values, predictions):
        return [t - p for t, p in zip(true_values, predictions)]

    @staticmethod
    def is_regression():
        return True


class GradientBoostingBinaryClassification(GradientBoostingTask):
    @staticmethod
    def create_default_model_friedman(data: Dataset):
        """
        The optimal initial value is log[(1 + average) / (1 - average)] / 2,
        where the targets in the data are +-1.
        :param data:
        :return:
        """
        tree = create_constant_tree(GradientBoostingTask.heuristic, data)
        regression_stats = NodeStatisticsRegression()
        regression_stats.add_examples(data.get_target_data())
        regression_stats.create_predictions()
        average = regression_stats.get_prediction()
        if abs(average - 1.0) < 10**-10:
            prediction = float('inf')  # TODO: ?
        elif abs(average + 1.0) < 10**-10:
            prediction = float('-inf')  # TODO: ?
        else:
            prediction = log((1 + average) / (1 - average)) / 2
        tree.root_node.get_stats().set_prediction(prediction)
        return tree

    @staticmethod
    def create_default_model(data: Dataset):
        return GradientBoostingRegression.create_default_model(data)

    @staticmethod
    def minus_partial_derivative_friedman(true_values, predictions):
        ans = []
        for t, p in zip(true_values, predictions):
            try:
                ans.append(2 * t / (1 + exp(2 * t * p)))
            except OverflowError:
                ans.append(t * 10**-10)  # t * float('inf')) TODO: ?
        return ans

    @staticmethod
    def minus_partial_derivative(true_values, predictions):
        return GradientBoostingRegression.minus_partial_derivative(
            true_values, predictions)

    @staticmethod
    def is_classification():
        return True

    @staticmethod
    def is_binary_classification():
        return True


class GradientBoostingMulticlassClassification(GradientBoostingTask):
    @staticmethod
    def create_default_model_friedman(data: Dataset):
        """
        Default model returns zero. We could leave it out, but technically it may be better to do this.
        :param data:
        :return:
        """
        tree = create_constant_tree(GradientBoostingTask.heuristic, data)
        tree.root_node.get_stats().set_prediction(0.0)
        return tree

    @staticmethod
    def create_default_model(data: Dataset):
        return GradientBoostingRegression.create_default_model(data)

    @staticmethod
    def minus_partial_derivative_friedman(true_values, predictions):
        examples = len(predictions[0])
        classes = len(predictions)
        predictions_exp = [[exp(p) for p in ps]
                           for ps in predictions]  # TODO: numpy? :)
        normalizers = [
            sum(predictions_exp[k][i] for k in range(classes))
            for i in range(examples)
        ]
        prob = [[
            predictions_exp[k][i] / normalizers[i] for i in range(examples)
        ] for k in range(classes)]
        return [[true_values[k][i] - prob[k][i] for i in range(examples)]
                for k in range(classes)]

    @staticmethod
    def minus_partial_derivative(true_values, predictions):
        return [
            GradientBoostingRegression.minus_partial_derivative(t, p)
            for t, p in zip(true_values, predictions)
        ]

    @staticmethod
    def is_classification():
        return True

    @staticmethod
    def is_multiclass_classification():
        return True


class GradientBoosting(TreeEnsemble):
    friedman = False
    binary_classification = GradientBoostingBinaryClassification
    multi_class_classification = GradientBoostingMulticlassClassification
    regression = GradientBoostingRegression

    def __init__(self,
                 nb_trees_to_build=100,
                 shrinkage=1.0,
                 optimize_step_size=True,
                 step_size=1.0,
                 chosen_examples=1.0,
                 random_seed=112,
                 **tree_parameters):
        self.nb_trees = nb_trees_to_build
        self.shrinkage = shrinkage
        self.optimize_step_size = optimize_step_size
        self.step_sizes = self.compute_step_sizes(step_size)
        self.chosen_examples = chosen_examples
        self.ensemble_random = EnsembleRandomGenerator(random_seed)
        self.tree_parameters = tree_parameters
        self.task = None
        self.alphas = 0
        self.trees = []  # type: List[DecisionTree]
        self.trees_per_class = []  # type: List[List[DecisionTree]]
        self.class_dictionary = {}

    def compute_step_sizes(self, step_size):
        try:
            return [s for s in step_size]
        except TypeError:
            return [step_size] * self.nb_trees

    @staticmethod
    def find_task(target_data: List[Datum]):
        data_type = type(target_data[0].get_target())
        if data_type == float or data_type == int:
            return GradientBoosting.regression
        elif data_type == str:
            c = len(get_all_target_values(target_data))
            if c == 2:
                return GradientBoosting.binary_classification
            elif c > 2:
                return GradientBoosting.multi_class_classification
            else:
                raise WrongValueException(
                    "Weird number of classes: {}".format(c))
        else:
            raise WrongValueException(
                "Wrong target type: {}".format(data_type))

    def build(self, input_data: Dataset):
        # find task
        self.task = GradientBoosting.find_task(input_data.get_target_data())
        if self.task in [
                GradientBoosting.binary_classification,
                GradientBoosting.regression
        ]:
            self.build_helper1(input_data)
        elif self.task in [GradientBoosting.multi_class_classification]:
            self.build_helper2(input_data)

    def build_helper1(self, input_data: Dataset):
        # preprocess data
        data, class_dictionary = self.preprocess(input_data)
        true_values = [datum.get_target() for datum in data.get_target_data()]
        self.class_dictionary = class_dictionary
        # build default model
        self.trees.append(self.task.create_default_model(data))
        current_predictions = self.trees[-1].predict_all(
            data.get_target_data())
        # build boosted trees
        for t in range(self.nb_trees):
            ys = self.task.minus_partial_derivative(true_values,
                                                    current_predictions)
            modified_data = self.modify_dataset(data, ys)
            print("Building tree {}".format(t + 1))
            self.tree_parameters[
                'random_seed'] = self.ensemble_random.next_tree_seed()
            self.tree_parameters['heuristic'] = HeuristicVariance()
            self.trees.append(DecisionTree(**self.tree_parameters))
            self.trees[-1].build(modified_data)
            self.task.modify_tree(self.shrinkage, self.step_sizes[t],
                                  self.optimize_step_size, self.trees[-1])
            GradientBoosting.update_current_predictions(
                self.trees[-1], current_predictions, data.get_target_data())

    def build_helper2(self, input_data: Dataset):
        # preprocess data
        datasets, class_dictionary = self.preprocess(input_data)
        k = len(datasets)
        true_values = [[
            datum.get_target() for datum in data.get_target_data()
        ] for data in datasets]
        self.class_dictionary = class_dictionary
        # build default model, for each class
        self.trees_per_class.append([])
        current_predictions = []
        for i in range(k):
            self.trees_per_class[-1].append(
                self.task.create_default_model(datasets[i]))
            current_predictions.append(self.trees_per_class[-1][i].predict_all(
                datasets[i].get_target_data()))
        # build boosted trees, for each class
        for t in range(self.nb_trees):
            ys = self.task.minus_partial_derivative(true_values,
                                                    current_predictions)
            modified_datasets = []
            self.trees_per_class.append([])
            for i in range(k):
                modified_datasets.append(
                    self.modify_dataset(datasets[i], ys[i]))
                print("Building tree {} for class {}".format(t + 1, i + 1))
                self.tree_parameters[
                    'random_seed'] = self.ensemble_random.next_tree_seed()
                self.tree_parameters['heuristic'] = HeuristicVariance()
                self.trees_per_class[-1].append(
                    DecisionTree(**self.tree_parameters))
                self.trees_per_class[-1][i].build(modified_datasets[i])
                self.task.modify_tree(self.shrinkage, self.step_sizes[t],
                                      self.optimize_step_size,
                                      self.trees_per_class[-1][i])
                GradientBoosting.update_current_predictions(
                    self.trees_per_class[-1][i], current_predictions[i],
                    datasets[i].get_target_data())

    @staticmethod
    def update_current_predictions(tree, current_predictions,
                                   data: List[Datum]):
        for i, datum in enumerate(data):
            current_predictions[i] += tree.predict(datum)

    def preprocess(self, data: Dataset):
        """
        The statistics of the data are changed to boosting statistics.
        For classification, targets are converted into numeric values.
        :param data:
        :return:
        """
        if self.task.is_binary_classification():
            # convert the two target names into +-1 for friedman and 0/1 otherwise
            dictionary = {}
            for target_tuple in data:
                t = target_tuple.get_target()
                if t not in dictionary:
                    value = len(
                        dictionary
                    ) * 2 - 1 if GradientBoosting.friedman else len(dictionary)
                    dictionary[t] = value
            assert len(dictionary) == 2
            new_target_values = []  # type: List[Datum]
            for datum in data:
                new_target_values.append(
                    Datum(datum.get_descriptive(),
                          dictionary[datum.get_target()], datum.get_weight(),
                          datum.identifier))
            if GradientBoosting.friedman:
                new_statistics = NodeStatisticsBinaryClassificationBoosting()
            else:
                new_statistics = NodeStatisticsRegressionBoosting()
            new_statistics.add_examples(new_target_values)
            return Dataset(settings=data.settings,
                           data_file=data.data_file,
                           descriptive_relations=data.get_descriptive_data(),
                           target_data=new_target_values,
                           statistics=new_statistics), {
                               y: x
                               for x, y in dictionary.items()
                           }
        elif self.task.is_multiclass_classification():
            # convert to k datasets
            dictionary = {}
            for target_tuple in data:
                t = target_tuple.get_target()
                if t not in dictionary:
                    value = len(dictionary)
                    dictionary[t] = value
            k = len(dictionary)
            # assert k > 2
            new_target_values = [[]
                                 for _ in range(k)]  # type: List[List[Datum]]
            for datum in data:
                for i in range(k):
                    new_target_values[i].append(
                        Datum(datum.get_descriptive(), 0.0, datum.get_weight(),
                              datum.identifier))
            if GradientBoosting.friedman:
                new_statistics = NodeStatisticsMulticlassClassificationBoosting(
                    k)
            else:
                new_statistics = NodeStatisticsRegressionBoosting()
            dataset_params = {
                'settings': data.settings,
                'descriptive_relations': data.get_descriptive_data(),
                'statistics': new_statistics
            }
            datasets = []
            for i in range(k):
                dataset_params['target_data'] = new_target_values[i]
                dataset_params['statistics'] = new_statistics.get_copy()
                dataset_params['data_file'] = data.data_file
                datasets.append(Dataset(**dataset_params))
            for i, datum in enumerate(data):
                non_zero = dictionary[datum.get_target()]
                datasets[non_zero].get_target_data()[i].set_target(1.0)
            for dataset in datasets:
                dataset.statistics.add_examples(dataset.get_target_data())
            return datasets, {y: x for x, y in dictionary.items()}
        elif self.task.is_regression():
            new_statistics = NodeStatisticsRegressionBoosting()
            new_statistics.add_examples(data.get_target_data())
            return Dataset(settings=data.settings,
                           data_file=data.data_file,
                           descriptive_relations=data.get_descriptive_data(),
                           target_data=data.get_target_data(),
                           statistics=new_statistics), None
        else:
            raise WrongValueException("Wrong task: {}".format(self.task))

    def modify_dataset(self, data: Dataset, ys):
        # update targets
        new_target_data = []
        for datum, y in zip(data.get_target_data(), ys):
            new_datum = Datum(datum.get_descriptive(), y, datum.get_weight(),
                              datum.identifier)
            assert new_datum.get_weight() == 1
            new_target_data.append(new_datum)
        # subsample
        if self.chosen_examples < 1.0:
            n = len(new_target_data)
            k = int(n * self.chosen_examples)
            random.seed(self.ensemble_random.next_sample_rows_seed())
            chosen_indices = random.sample(range(n), k=k)
            new_target_data = [new_target_data[i] for i in chosen_indices]
        return Dataset(settings=data.settings,
                       data_file=data.data_file,
                       descriptive_relations=data.get_descriptive_data(),
                       target_data=new_target_data,
                       statistics=data.get_copy_statistics())

    def compute_ranking(self, ranking_type):
        feature_ranking = EnsembleRanking({}, {}, ranking_type, self.nb_trees)
        for i in range(self.nb_trees):
            if self.task in [
                    GradientBoosting.binary_classification,
                    GradientBoosting.regression
            ]:
                trees = [self.trees[i + 1]]
            elif self.task in [GradientBoosting.multi_class_classification]:
                trees = self.trees_per_class[i + 1]
            else:
                raise WrongValueException("Unknown task: {}.".format(
                    self.task))
            scores = [
                feature_ranking.compute_tree_contribution(tree)
                for tree in trees
            ]
            average_scores = []
            for j in range(2):
                average_scores.append(
                    average_of_dictionaries([pair[j] for pair in scores]))
            feature_ranking.update_attributes(average_scores[0],
                                              average_scores[1], i)
        # We do not normalize the ranking ... feature_ranking.normalize()
        return feature_ranking

    def predict(self, d: Datum, nb_trees=None):
        """
        Predict the target values of datum d.
        :param d:
        :param nb_trees: Number of trees that are used for prediction. Maximal value is the value of the
        argument nb_trees_to_build from the constructor. If set to 0, only default predictions are used.
        :return: Prediction for datum d.
        """
        if nb_trees is None:
            nb_trees = self.nb_trees + 1
        else:
            nb_trees += 1
        if self.task in [
                GradientBoosting.binary_classification,
                GradientBoosting.regression
        ]:
            return self.predict_helper1(d, nb_trees)
        elif self.task in [GradientBoosting.multi_class_classification]:
            return self.predict_helper2(d, nb_trees)
        else:
            raise NotImplementedError(":D")

    def predict_helper1(self, d: Datum, nb_trees):
        predictions_stats = []  # type: List[NodeStatistics]
        for tree in self.trees[:nb_trees]:
            predictions_stats.append(tree.predict(d, True))
        # statistics_class = predictions_stats[0].__class__
        prediction = 0.0
        for s in predictions_stats:
            prediction += s.get_prediction()
        if self.task == GradientBoosting.binary_classification:
            numeric_classes = [-1, 1] if GradientBoosting.friedman else [0, 1]
            original_classes = [
                self.class_dictionary[n] for n in numeric_classes
            ]
            if GradientBoosting.friedman:
                p_numeric_classes = []
                for n in numeric_classes:
                    try:
                        p_numeric_classes.append(
                            1 / (1 + exp(-2 * n * prediction)))
                    except OverflowError:
                        p_numeric_classes.append(0.0)
            else:
                p_numeric_classes = [1 - prediction, prediction]
            return original_classes[arg_max(p_numeric_classes)]
        elif self.task == GradientBoosting.regression:
            return prediction
        else:
            return WrongValueException("Wrong task: {}".format(self.task))

    def predict_helper2(self, d: Datum, nb_trees):
        k = len(self.class_dictionary)
        predictions_stats = []  # type: List[List[NodeStatistics]]
        for tree_ind in range(nb_trees):
            predictions_stats.append([])
            for class_ind in range(k):
                tree = self.trees_per_class[tree_ind][class_ind]
                predictions_stats[-1].append(tree.predict(d, True))
        prediction = [0.0 for _ in range(k)]
        for s in predictions_stats:
            for class_ind in range(k):
                prediction[class_ind] += s[class_ind].get_prediction()
        if self.task == GradientBoosting.multi_class_classification:
            numeric_classes = list(range(k))
            original_classes = [
                self.class_dictionary[n] for n in numeric_classes
            ]
            if GradientBoosting.friedman:
                # exp is monotonic, so we do not need it for arg max, but nevertheless ...
                prediction = [exp(p) for p in prediction]
            prediction_sum = sum(prediction)
            p_numeric_classes = [p / prediction_sum for p in prediction]
            return original_classes[arg_max(p_numeric_classes)]
        else:
            return WrongValueException("Wrong task: {}".format(self.task))

    def print_model(self, file_name):
        f = open(file_name, "w")
        if self.class_dictionary:
            print("Class encoding: {}".format(self.class_dictionary), file=f)
        if self.task in [
                GradientBoosting.binary_classification,
                GradientBoosting.regression
        ]:
            for i, tree in enumerate(self.trees):
                print("Tree {}:".format(i), file=f)
                print(str(tree), file=f)
                print("", file=f)
        elif self.task in [GradientBoosting.multi_class_classification]:
            for i, trees in enumerate(self.trees_per_class):
                for c, tree in enumerate(trees):
                    print("Tree {} for class {}:".format(i, c + 1), file=f)
                    print(str(tree), file=f)
                    print("", file=f)
        f.close()
