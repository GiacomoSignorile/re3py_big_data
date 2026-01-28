from typing import Iterable
from .core.aggregators import *
from .core.comparators import *
from .core.tree_node_split import BinarySplit
from .core.heuristic import Heuristic
from ..data.data_and_statistics import *
from ..data.task_settings import Settings
from .core.variables import *
from ..data.relation import Relation
from ..utilities.my_exceptions import WrongValueException
from ..utilities.my_utils import *
import itertools
import random
from .predictive_model import PredictiveModel
from time import perf_counter as clock  # Python 3.8+ compatibility
import math
import copy
from .core.tree_node_split import TEST_VALUE_MEMO
# from my_memo import used_comp_memo
import subprocess
from .core.communicate_with_java import send_data, send_variables, compute_test_values
from py4j.java_gateway import JavaGateway, GatewayParameters


class TreeNode:
    positive_branch = 0
    negative_branch = 1

    def __init__(self, description, parent, children,
                 binary_split: Union[BinarySplit, None],
                 stats: Union[NodeStatistics, None], depth: int):
        self.description = description
        self.parent = parent  # type: 'TreeNode'
        self.children = children
        self.split = binary_split  # type: BinarySplit
        self.stats = stats  # type: NodeStatistics
        self.depth = depth

    def get_parent(self) -> 'TreeNode':
        return self.parent

    def get_split(self):
        return self.split

    def set_depth(self, d):
        self.depth = d

    def get_depth(self):
        return self.depth

    def set_split(self, s):
        self.split = s

    def set_parent(self, other):
        self.parent = other

    def get_child(self, i):
        return self.children[i]

    def get_children(self):
        return self.children

    def add_child(self, other):
        self.add_children([other])

    def add_children(self, others):
        if self.children is None:
            self.children = []
        self.children += others

    def is_leaf(self):
        return len(self.children) == 0

    def get_stats(self):
        return self.stats

    def set_stats(self, s):
        self.stats = s


class DecisionTree(PredictiveModel):
    root_node_depth = 1
    eps = 10**-10
    left_child_indicator = "0"
    right_child_indicator = "1"
    root_indicator = "root"

    square_root = "sqrt"
    log2 = "log"
    allowed_n_tests = [square_root, log2]

    def __init__(
            self,
            heuristic=None,
            statistics=None,
            root_node=None,
            max_number_internal_nodes=float('inf'),
            max_number_atom_tests=1,
            allowed_atom_tests=None,
            allowed_aggregators: Union[None, Iterable[str]] = None,
            max_depth=float("inf"),
            minimal_examples_in_leaf=1,
            max_number_of_evaluated_tests_per_node=float("inf"),
            max_relative_number_of_evaluated_tests_per_node: Union[float,
                                                                   str] = 1.0,
            random_seed=2718281828,
            java_port: Union[None, int] = None,
            is_outer_java=False,
            longest_atom_test_chain=4,
            class_weights: Union[None, Dict[str, float]] = None,
            per_class_bootstrap=False,
            only_existential=False,
            minimal_impurity=10**-16):
        self.heuristic = Heuristic() if heuristic is None else heuristic
        self.target_data_stat = statistics
        self.max_number_internal_nodes = max_number_internal_nodes
        self.max_number_atom_tests = max_number_atom_tests
        self.allowed_atom_tests = {} if allowed_atom_tests is None else allowed_atom_tests  # the structured version
        self.allowed_aggregators = set(
        ) if allowed_aggregators is None else set(allowed_aggregators)
        self.max_depth = max_depth
        self.minimal_examples_in_leaf = minimal_examples_in_leaf
        self.max_number_of_evaluated_tests_per_node = max_number_of_evaluated_tests_per_node
        self.max_relative_number_of_evaluated_tests_per_node = max_relative_number_of_evaluated_tests_per_node
        self.relative_number_tests_sanity_check()
        self.random_seed = random_seed
        self.java_port = java_port
        self.is_outer_java = is_outer_java
        self.longest_atom_test_chain = longest_atom_test_chain
        self.class_weights = class_weights
        self.per_class_bootstrap = per_class_bootstrap
        self.only_existential = only_existential
        self.update_allowed_aggregates()
        self.minimal_impurity = minimal_impurity  # relative

        self.root_node = root_node  # type: Union['TreeNode', None]
        self.target_relation_description = None
        # tree building fields
        self.target_relation_variables = []  # type: List[Variable]
        self.descriptive_data = {}  # type: Dict[str, Relation]
        self.var_count = {"XY": 0, "C": 0}  # TODO: less hard-coded?:)
        self.temp_var_count = 0
        self.current_number_internal_nodes = 0
        self.all_variables = {}
        self.induce_tree_time = 0.0
        self.statistics_time = 0.0
        self.get_test_value_time = 0.0
        self.split_eval_time = 0.0
        self.set_example_values_time = 0.0
        self.find_values_time = 0.0
        self.nominal_tests = 0
        self.nominal_tests_time = 0
        self.numeric_tests = 0
        self.numeric_tests_time = 0

        self.p = None
        self.wrapper = None
        self.client = None
        self.gateway = None

    def print_times(self):
        times = [
            self.induce_tree_time, self.statistics_time,
            self.get_test_value_time, self.split_eval_time,
            self.set_example_values_time, self.find_values_time,
            self.nominal_tests, self.nominal_tests_time, self.numeric_tests,
            self.numeric_tests_time
        ]
        names = [
            "induce", "statistics", "test value", "eval split",
            "example values", "find values", "nominal tests", "nom. t. time",
            "numeric tests", "num t. time"
        ]
        for time, name in zip(times, names):
            print("{: <14}:".format(name), time)
        print()

    def __str__(self):
        def helper(node: TreeNode):
            branches_names = ["YES", "NO"]
            spaces = "  " * node.get_depth()
            if node.is_leaf():
                return "{}{}".format(spaces, node.get_stats())
            else:
                parts = [
                    "{}IF {}:".format(spaces,
                                      node.split.__str__(self.all_variables))
                ]
                for b_name, child in zip(branches_names, node.children):
                    parts.append("{}{}".format(spaces, b_name))
                    parts.append(helper(child))
                return "\n".join(parts)

        header = "Tree for {}:".format(self.target_relation_description)
        tree = helper(self.root_node)
        return "\n".join([header, tree])

    def __iter__(self):
        """
        Iterates over the nodes in the tree: yields root, then 'recursively' visits the left and the right subtree.
        :return:
        """
        if self.root_node is not None:
            stack = [self.root_node]
            while stack:
                node = stack.pop()
                yield node
                stack += node.get_children()[::-1]

    def print_model(self, file_name):
        f = open(file_name, "w")
        print(str(self), file=f)
        f.close()

    def get_absolute_number_tests_from_relative(self, nb_tests):
        n_rel = self.max_relative_number_of_evaluated_tests_per_node
        if isinstance(n_rel, str):
            if n_rel == DecisionTree.square_root:
                return round(math.sqrt(nb_tests))
            elif n_rel == DecisionTree.log2:
                return round(math.log2(nb_tests))
            else:
                raise ValueError("Wrong function: {}".format(n_rel))
        elif isinstance(n_rel, float):
            return round(n_rel * nb_tests)
        else:
            raise ValueError(
                "Wrong relative number of tests: {}.".format(n_rel))

    def relative_number_tests_sanity_check(self):
        if isinstance(self.max_relative_number_of_evaluated_tests_per_node,
                      str):
            if self.max_relative_number_of_evaluated_tests_per_node not in DecisionTree.allowed_n_tests:
                message = "Wrong number of tests: {}. If string, it should be an element of {}"
                raise ValueError(
                    message.format(
                        self.max_relative_number_of_evaluated_tests_per_node,
                        DecisionTree.allowed_n_tests))
        elif isinstance(self.max_relative_number_of_evaluated_tests_per_node,
                        float):
            if not (0.0 <= self.max_relative_number_of_evaluated_tests_per_node
                    <= 1):
                message = "Wrong relative number of tests: {}. If float, should be in the interval (0, 1]"
                raise ValueError(
                    message.format(
                        self.max_relative_number_of_evaluated_tests_per_node))
        else:
            raise ValueError(
                "Relative number of tests should be string or float.")

    def chosen_tests(self, tests_generator, current_vars_per_type):
        nb_tests = 0
        nb_chains = 0
        for jj, chain in enumerate(tests_generator):
            # if jj % 10**2 == 0:
            #     print("-", end="")
            # if jj % 10**4 == 0:
            #     print()
            nb_chains += 1
            _, relation_chain, aggregator_chains, _ = chain
            # print(relation_chain)
            num_aggregator_chains = len(list(aggregator_chains))
            _, _, _, _, c_num = self.create_example_and_chains(
                relation_chain, current_vars_per_type)
            nb_tests += num_aggregator_chains * c_num
        # print()
        k_proportion = self.get_absolute_number_tests_from_relative(nb_tests)
        k_absolute = self.max_number_of_evaluated_tests_per_node
        k = min(k_proportion,
                k_absolute)  # Select the more restrictive criterion
        k = max(k, 1)  # but at least one
        k = min(k, nb_tests)  # but at most all tests
        return set(random.sample(range(nb_tests), k=k)), nb_tests, nb_chains

    def generate_next_var_name(self, first_letter):
        key = None
        if first_letter == "C":
            key = "C"
        elif first_letter in "XY":
            key = "XY"
        c = self.var_count[key]
        self.var_count[key] += 1
        return "{}{}".format(first_letter, c)

    def generate_temp_var_name(self, first_letter):
        return self.generate_temp_var_names(first_letter, [1])[0]

    def generate_temp_var_names(self, first_letter, offsets):
        names = [
            "{}{}".format(first_letter, self.temp_var_count - offset)
            for offset in offsets
        ]
        if names:
            self.temp_var_count -= max(offsets)
        return names

    def reset_temp_var_count(self):
        self.temp_var_count = 0

    @staticmethod
    def get_first_var_letter(object_type, is_input):
        if object_type == Settings.atom_test_type_constant:
            return "C"
        elif is_input:
            return "X"
        else:
            return "Y"

    def initialize_statistics(self, node: TreeNode, data: List[Datum]):
        t0 = clock()
        statistics_class = self.target_data_stat.__class__
        initial_stats = statistics_class.construct_from_parent(
            self.target_data_stat)
        initial_stats.add_examples(data)
        node.set_stats(initial_stats)
        variability = self.heuristic.compute_variability(node.get_stats())
        node.get_stats().set_variability(variability)
        t1 = clock()
        self.statistics_time += t1 - t0

    def build(self, data: Dataset):
        t0 = clock()
        random.seed(self.random_seed)
        self.target_data_stat = data.get_copy_statistics()
        target_relation = data.get_target_relation()  # no examples - ok?
        self.descriptive_data = data.get_descriptive_data(
        )  # type: Dict[str, Relation]
        target_data = data.get_target_data()  # type: List[Datum]
        if BinarySplit.use_memo:
            DecisionTree.populate_dict(target_data)
        current_vars_per_type = [{}]  # type: List[Dict[str, Set[Variable]]]
        target_var_names = []
        self.target_relation_variables = []  # type: List[Variable]
        for object_type in target_relation.get_types()[:-1]:
            var_name = self.generate_next_var_name(
                DecisionTree.get_first_var_letter(object_type, True))
            assert not var_name.startswith("C")
            new_var = VariableVariable(var_name, object_type, None)
            if object_type not in current_vars_per_type[-1]:
                current_vars_per_type[-1][object_type] = set()
            current_vars_per_type[-1][object_type].add(new_var)
            self.target_relation_variables.append(new_var)
            target_var_names.append(new_var.get_name())

        self.target_relation_description = "{}, {}".format(
            target_relation.get_name(), target_var_names)
        all_variable_names = set(target_var_names)
        self.root_node = TreeNode(DecisionTree.root_indicator, None, [], None,
                                  None, DecisionTree.root_node_depth)
        # initial statistics
        self.initialize_statistics(self.root_node, target_data)

        if self.java_port is not None:
            self.java_on()

        if self.java_port is not None:
            send_data(data, self.client, self.wrapper)

        # manipulate target data
        self.target_data_induction_preparation(target_data)
        self.build_helper(target_data, self.root_node, current_vars_per_type,
                          target_var_names, all_variable_names)
        # un-manipulate target data
        self.reverse_target_data_induction_preparation(target_data)

        for v in self.target_relation_variables:
            assert v.can_vary()
            v.unset_value()
        for v in self.target_relation_variables:
            self.all_variables[v.get_name()] = v
        t1 = clock()
        self.induce_tree_time = t1 - t0
        self.print_times()
        # s = max(1, sum(used_comp_memo))
        # print("Memo vs. compute: {:.4f} : {:.4f}; all: {}".format(used_comp_memo[0] / s,
        #                                                           used_comp_memo[1] / s,
        #                                                           s))
        # used_comp_memo[0] = 0
        # used_comp_memo[1] = 0

        if self.java_port is not None:
            self.java_off()

    def java_on(self):
        if not self.is_outer_java:
            # open server
            this_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(this_dir, "core", "speedUp.jar")
            if not os.path.exists(path):
                path = "speedUp.jar"
            self.p = subprocess.Popen(r"java -jar {} {}".format(
                path, self.java_port),
                                      shell=True)
            for counter in range(10**8):
                _ = 21 + 21
        gateway_parameters = GatewayParameters(port=self.java_port)
        self.gateway = JavaGateway(gateway_parameters=gateway_parameters)
        self.wrapper = self.gateway.entry_point.get_wrapper()
        self.client = self.gateway._gateway_client

    def java_off(self):
        if not self.is_outer_java:
            self.gateway.shutdown()
            self.p.kill()
        self.wrapper = None
        self.client = None
        self.gateway = None
        self.p = None

    def update_allowed_aggregates(self):
        if self.only_existential:
            self.allowed_aggregators = {COUNT.get_name()}
            print(
                "Existential tests only ==> The allowed aggregates changed to {}"
                .format(self.allowed_aggregators))

    def build_helper(self, target_data: List[Datum], current_node: TreeNode,
                     current_vars_per_type, target_relation_vars,
                     all_variable_names: Set[str]):
        # print("Current vars before inducing", current_node.description)
        # print("cur_ver_per_type, all_names", current_vars_per_type, all_variable_names)
        print("{}Building node on depth {}".format(
            "  " * current_node.get_depth(), current_node.get_depth()))
        current_var_names = [{
            t: {v.get_name()
                for v in vs}
            for t, vs in d.items()
        } for d in current_vars_per_type]
        # print("curr var names", current_var_names)
        # find a split
        bs = BinarySplit([], None, None, True, None)
        best_score = BinarySplit.worst_split_score
        best_configuration = (None, None, None, None, None, None)
        if self.should_try_find_a_split(current_node, target_data):
            parents_test = []
            node = current_node
            while node.parent is not None:
                if node == node.get_parent().children[0]:
                    parents_test = node.get_parent().get_split().get_test()
                    break
                node = node.get_parent()
            attributes_counting = self.generate_possible_attributes(
                copy.deepcopy(current_var_names), parents_test,
                target_relation_vars)
            self.reset_temp_var_count()
            attributes = self.generate_possible_attributes(
                current_var_names, parents_test, target_relation_vars)

        else:
            attributes_counting = iter([])
            attributes = iter([])
        chosen_attributes, all_attributes_computed, all_chains_computed = self.chosen_tests(
            attributes_counting, current_vars_per_type)
        # print("{}Attributes generated: {}".format("  " * current_node.get_depth(), all_attributes_computed))
        all_attributes_counted = 0
        all_chains_counted = 0
        for chain in attributes:
            all_chains_counted += 1
            # print(".", end="")
            # if all_chains_counted % 100 == 0:
            #     print("{}".format("  " * current_node.get_depth()), end="")
            starting_index, relation_chain, aggregator_chains, fresh_vars = chain
            # print("{}relation chain: {}".format("  " * current_node.get_depth(), relation_chain))
            nb_fresh_vars = len(fresh_vars[0])
            fresh_indices = fresh_vars[1]
            # print(relation_chain)
            example, rc_modified, c_values, c_var_names, c_num = self.create_example_and_chains(
                relation_chain, current_vars_per_type)
            a_ch = list(aggregator_chains)
            # print(a_ch)
            known_unknown = None
            temp_counter = 0
            for c_vs in c_values:
                filtered_agg_chains = []
                filtered_output_types = []
                for i, agg_chain in enumerate(a_ch):
                    if all_attributes_counted in chosen_attributes:
                        filtered_agg_chains.append(agg_chain[0])
                        filtered_output_types.append(agg_chain[1])
                    all_attributes_counted += 1
                    temp_counter += 1
                # print("   csv -->", c_vs)
                for c_name, c_v in zip(c_var_names, c_vs):
                    example[c_name].set_value(c_v)
                if BinarySplit.use_memo or known_unknown is None:
                    # unset target variables
                    for init_var_name in target_relation_vars:
                        example[init_var_name].unset_value()
                    r_key, a_keys, known_unknown = DecisionTree.test_values_memo_keys(
                        example, rc_modified, filtered_agg_chains)
                else:
                    r_key, a_keys = None, []
                t0 = clock()

                if self.java_port is not None:
                    # do stuff here
                    send_variables(example, self.client, self.wrapper)
                    all_test_values = compute_test_values(
                        target_data, target_relation_vars, rc_modified,
                        filtered_agg_chains, r_key, a_keys, nb_fresh_vars,
                        fresh_indices, known_unknown, self.client,
                        self.wrapper)
                else:
                    all_test_values = []
                    for jj, datum in enumerate(target_data):
                        u0 = clock()
                        for init_var_name, train_value in zip(
                                target_relation_vars, datum.get_descriptive()):
                            example[init_var_name].set_value(train_value)
                        u1 = clock()
                        self.set_example_values_time += u1 - u0
                        u0 = clock()
                        all_test_values.append(
                            bs.get_test_values(example, rc_modified,
                                               filtered_agg_chains, r_key,
                                               a_keys, datum.identifier,
                                               nb_fresh_vars, fresh_indices,
                                               known_unknown))
                        u1 = clock()
                        self.find_values_time += u1 - u0
                        # if jj % 10 == 0:
                        #     print(".", end="")

                # n_target_examples = len(target_data)
                # assert n_target_examples == len(all_test_values1) == len(all_test_values2)
                # for i, v1, v2 in zip(range(n_target_examples), all_test_values1, all_test_values2):
                #     if v1 != v2:
                #         print(i, v1, v2, target_data[i])
                #         raise ValueError("... :)")
                # print()

                t1 = clock()
                self.get_test_value_time += t1 - t0
                t0 = clock()
                score, configuration = self.evaluate_candidate_splits(
                    current_node, all_test_values, target_data,
                    current_vars_per_type[0], filtered_output_types)
                t1 = clock()
                self.split_eval_time += t1 - t0
                if BinarySplit.is_better_than_previous(score, best_score):
                    best_score = score
                    a_chain_ind, comparator, theta, partition, is_variable_free = configuration
                    best_configuration = (rc_modified,
                                          filtered_agg_chains[a_chain_ind],
                                          c_vs, c_var_names, comparator, theta,
                                          partition, is_variable_free,
                                          starting_index)
            # unset constants
            for c_name in c_var_names:
                example[c_name].unset_value()
            # unset target variables
            for init_var_name in target_relation_vars:
                example[init_var_name].unset_value()
            # sanity check
            if temp_counter != c_num * len(a_ch):
                print(temp_counter, c_num, len(a_ch))
                print(example, rc_modified, c_values, c_var_names, c_num, a_ch)
                print(relation_chain)
                raise WrongValueException("Wrong value of counted attributes!")
        # sanity check
        if all_attributes_computed != all_attributes_counted or all_chains_computed != all_chains_counted:
            message = "\nPredicted number of attributes: {} Number of attributes counted: {}\n" \
                      "Predicted number of chains: {} Counted number of chains: {}"
            raise WrongValueException(
                message.format(all_attributes_computed, all_attributes_counted,
                               all_chains_computed, all_chains_counted))
        # create internal node or leaf
        if BinarySplit.is_better_than_previous(best_score,
                                               BinarySplit.worst_split_score):
            self.current_number_internal_nodes += 1
            r_chain, a_chain, c_vs, c_var_names, comparator, theta, partition, is_variable_free, starting_index = \
                best_configuration
            # print("---> r chain, c_var names", r_chain, c_var_names, sep="\n")
            r_chain_fresh_part = len(r_chain) - starting_index + 1
            split_test = []
            # [d1, ...], d2: {type1: {freshVariable1, ...}, ...}
            fresh_variables_chains = [{} for _ in range(r_chain_fresh_part)]
            fresh_variables = {}  # Union of the upper
            const_name_to_value = {n: v for n, v in zip(c_var_names, c_vs)}
            temp_to_fix_name = {}
            # find fresh variables, get rid of temporary names
            for i_relation, ((r, var_names),
                             a) in enumerate(zip(r_chain, a_chain)):
                new_names = []
                for var_name, var_type in zip(
                        var_names,
                        self.descriptive_data[r.get_name()].get_types()):
                    # not seen in previous chains?
                    condition1 = var_name not in all_variable_names
                    condition2 = int(var_name[1:]) < 0
                    if condition1 != condition2:
                        print(var_name, all_variable_names)
                        print(r_chain)
                        exit(-12345)
                    if condition1:
                        assert i_relation + 1 >= starting_index
                        fresh_variables_chain = fresh_variables_chains[
                            i_relation - (starting_index - 1)]
                        # not seen in this chain also?
                        if var_name not in temp_to_fix_name:
                            # we see this name for the first time --> rename from temp name
                            new_var_name = self.generate_next_var_name(
                                var_name[0])
                            # set the value of constant variables
                            is_const1 = var_name in const_name_to_value
                            is_const2 = var_name[0] == "C"
                            assert is_const1 == is_const2
                            value = const_name_to_value[
                                var_name] if is_const2 else None
                            new_var = create_new_variable(
                                new_var_name, var_type, value)
                            if var_type not in fresh_variables_chain:
                                fresh_variables_chain[var_type] = set()
                            fresh_variables_chain[var_type].add(new_var)
                            temp_to_fix_name[var_name] = new_var_name
                            fresh_variables[new_var_name] = new_var
                        else:
                            new_var_name = temp_to_fix_name[var_name]
                    else:
                        new_var_name = var_name
                    new_names.append(new_var_name)
                split_test.append((r, new_names, a))
            self.all_variables.update(fresh_variables)
            fresh_variables_names = set(fresh_variables.keys())
            self.reset_temp_var_count()
            # set the split
            bs = BinarySplit(split_test, comparator, theta, True,
                             is_variable_free)
            bs.add_fresh_variables(fresh_variables)
            current_node.set_split(bs)
            # branch frequencies
            nb_examples = target_data_weight(target_data)
            branch_frequencies = [
                target_data_weight(target_data, part) / nb_examples
                for part in partition
            ]
            current_node.get_stats().set_branch_frequencies(branch_frequencies)
            # create children etc.
            current_variables_children = [
                current_vars_per_type[:starting_index] +
                fresh_variables_chains, current_vars_per_type
            ]
            # print("variables before/fresh after inducing", current_node.description)
            # print(all_variable_names, fresh_variables_names)
            for i, part in enumerate(partition):
                assert i < 2
                label = current_node.description + ".{}".format(i)
                child = TreeNode(label, current_node, [], None, None,
                                 current_node.get_depth() + 1)
                current_node.add_child(child)
                target_data_child = [target_data[i] for i in part]
                self.initialize_statistics(child, target_data_child)
                current_variables_child = current_variables_children[i]
                all_variable_names_child = all_variable_names | fresh_variables_names if i == 0 else all_variable_names
                self.build_helper(target_data_child, child,
                                  current_variables_child,
                                  target_relation_vars,
                                  all_variable_names_child)
        else:
            current_node.get_stats().create_predictions()

    def should_try_find_a_split(self, current_node: TreeNode,
                                target_data: List[Datum]):
        if current_node.get_depth() >= self.max_depth:
            return False
        elif self.current_number_internal_nodes >= self.max_number_internal_nodes:
            return False
        elif target_data_weight(
                target_data
        ) <= 2 * self.minimal_examples_in_leaf - DecisionTree.eps:
            return False
        elif current_node.get_stats(
        ).variability < DecisionTree.eps * self.root_node.get_stats(
        ).variability:
            # print("Variability too low: ", self.heuristic.compute_variability(current_node.get_stats()))
            return False
        else:
            return True

    def evaluate_candidate_splits(self, parent, test_values, target_data,
                                  target_relation_vars, filtered_output_types):
        n_aggregators = len(test_values[0])
        best_score = BinarySplit.worst_split_score
        best_configuration = None
        for i in range(n_aggregators):
            xs = [x[i] for x in test_values]
            if filtered_output_types[i] == TYPE_NUMERIC:  # is numeric
                t0 = clock()
                score, comparator, theta, partition = self.find_best_numeric(
                    parent, xs, target_data)
                is_variable_free = True
                t1 = clock()
                self.numeric_tests += 1
                self.numeric_tests_time += t1 - t0
            else:  # is nominal: 'constant' or user defined type
                t0 = clock()
                output_type = filtered_output_types[i]
                if output_type in target_relation_vars:
                    var_candidates = sorted(
                        [v.name for v in target_relation_vars[output_type]])
                else:
                    var_candidates = []
                score, comparator, theta, partition, is_variable_free = self.find_best_nominal(
                    parent, xs, target_data, var_candidates, output_type)
                t1 = clock()
                self.nominal_tests_time += t1 - t0
            if BinarySplit.is_better_than_previous(score, best_score):
                best_score = score
                best_configuration = (i, comparator, theta, partition,
                                      is_variable_free)
        return best_score, best_configuration

    def find_best_nominal(self, parent: TreeNode, xs: List[str], target_data: List[Datum],
                          target_relation_vars: List[str], output_type: str, max_set_size=5) \
            -> Tuple[float, Comparator, Set[str], List[List[int]], bool]:
        if self.only_existential:
            raise ValueError(
                "Only existential tests cannot leave to nominal tests.")
        is_usual_nominal = Relation.is_nominal_type(output_type)
        if is_usual_nominal:
            different_values = sorted(set(
                xs))  # better than list, so that the order is always the same
            different_values_helper = []
            target_relation_var_indices = []
        else:
            different_values_helper = target_relation_vars
            different_values = list(range(len(different_values_helper)))
            target_relation_var_indices = [
                int(v[1:]) for v in target_relation_vars
            ]
        n = len(different_values)
        m = min(n, max_set_size)
        if is_usual_nominal:
            options = 2**(
                m - 1
            )  # Half of them suffice, empty set skipped later if necessary
        else:
            # Possible values are:
            # - value of (at least one of) the target variable(s)
            # - something else (including MODE_OF_EMPTY_LIST)
            # We implicitly skip all the subsets of the form {'something else', ...},
            # so we have to generate 2 ** m options (-1 for the empty one).
            options = 2**m
        if n > max_set_size:
            # message = "Warning: Nominal split: Too many subsets of a set with size = {}." \
            #           " Will evaluate {} randomly chosen ones."
            # print(message.format(n, options))
            subsets = random_subsets(different_values, options)
        else:
            subsets = subsets_of_list(different_values, options)
            next(subsets)  # skip the empty set
        best_score = BinarySplit.worst_split_score
        best_comparator = None
        best_subset = None
        best_partition = None
        generic_statistic = self.target_data_stat.__class__.construct_from_parent(
            parent.get_stats())
        left = []  # to prevent "unbound" warnings
        for left_list, _ in subsets:
            self.nominal_tests += 1
            if is_usual_nominal:
                left = set(left_list)
            partition = [[], []]  # type: List[List[int]]
            statistics = [
                generic_statistic.get_copy() for _ in range(len(partition))
            ]
            for datum_in, (test_value,
                           datum) in enumerate(zip(xs, target_data)):
                j = 1
                if is_usual_nominal:
                    if test_value in left:
                        j = 0
                else:
                    for chosen in left_list:
                        if datum.descriptive_part[target_relation_var_indices[
                                chosen]] == test_value:
                            j = 0
                            break
                partition[j].append(datum_in)
                statistics[j].add_example_during_split_eval(datum)
            # update all stats
            for sta in statistics:
                sta.after_split_evaluation_update()
            if self.is_split_valid(statistics):
                score = self.heuristic.evaluate_split(parent.get_stats(),
                                                      statistics)
            else:
                score = BinarySplit.worst_split_score
            if BinarySplit.is_better_than_previous(score, best_score):
                best_score = score
                best_partition = partition
                transpose_branches = DecisionTree.should_transpose_branches(
                    statistics)
                if transpose_branches:
                    # best_partition = best_partition[::-1]
                    best_comparator = DOES_NOT_CONTAIN
                else:
                    best_comparator = CONTAINS
                if is_usual_nominal:
                    best_subset = left  # right if transpose_branches else left
                else:
                    best_subset = {
                        different_values_helper[chosen]
                        for chosen in left_list
                    }
        return best_score, best_comparator, best_subset, best_partition, is_usual_nominal

    def find_best_numeric(self, parent: TreeNode, xs: List[float], target_data: List[Datum]) \
            -> Tuple[float, Comparator, float, List[List[int]]]:
        n = len(xs)
        sorted_indices = sorted(range(n), key=lambda t: xs[t])
        best_score = BinarySplit.worst_split_score
        best_comparator = SMALLER
        best_threshold = -float('inf')
        best_i = None
        best_partition = None
        transpose_branches = None
        generic_statistic = self.target_data_stat.__class__.construct_from_parent(
            parent.get_stats())
        statistics = [generic_statistic,
                      parent.get_stats().get_copy()
                      ]  # type: List[NodeStatistics]
        previous_value = xs[sorted_indices[0]]
        if self.only_existential:
            # The only possible test is x > 0, so the xs are converted into ones and zeros
            min_x, max_x = xs[sorted_indices[0]], xs[sorted_indices[-1]]
            if min_x < 0 or max_x == float("inf"):
                message = "Counting should result in numbers from [0, inf). Your range: [{}, {}]"
                raise ValueError(message.format(min_x, max_x))
            # modify: 0 --> 0 and > 0 --> 1 to ensure only one test below
            xs_modified = xs[:]
            for i in range(n):
                xs_modified[i] = 0 if xs[i] == 0 else 1
        else:
            xs_modified = xs

        for i in range(n):
            datum = target_data[sorted_indices[i]]
            x = xs_modified[sorted_indices[i]]
            if x > previous_value:
                if x < float('inf'):
                    if previous_value > float('-inf'):
                        threshold = previous_value + (
                            x - previous_value) / 2  # element of (previous, x)
                    else:
                        threshold = x - 21.21  # element of (-inf, x)
                else:
                    threshold = previous_value + 21.21  # element of (previous, inf)
                previous_value = x
                if self.is_split_valid(statistics):
                    score = self.heuristic.evaluate_split(
                        parent.get_stats(), statistics)
                else:
                    score = BinarySplit.worst_split_score
                if BinarySplit.is_better_than_previous(score, best_score):
                    best_score = score
                    best_threshold = threshold
                    best_i = i
                    transpose_branches = DecisionTree.should_transpose_branches(
                        statistics)
            statistics[TreeNode.
                       negative_branch].remove_example_during_split_eval(datum)
            statistics[TreeNode.positive_branch].add_example_during_split_eval(
                datum)
        if BinarySplit.is_better_than_previous(best_score,
                                               BinarySplit.worst_split_score):
            best_partition = [sorted_indices[:best_i], sorted_indices[best_i:]]
        if transpose_branches:
            best_comparator = BIGGER
            best_partition = best_partition[::-1]
        return best_score, best_comparator, best_threshold, best_partition

    def is_split_valid(self, stats: List[NodeStatistics]):
        for stat in stats:
            if stat.get_total_number_examples(
            ) <= self.minimal_examples_in_leaf - DecisionTree.eps:
                return False
        return True

    @staticmethod
    def should_transpose_branches(statistics: List[NodeStatistics]):
        weight_positive = statistics[0].get_total_number_examples()
        weight_negative = statistics[1].get_total_number_examples()
        return weight_positive < weight_negative

    def generate_possible_attributes(self, current_var_names,
                                     current_atom_tests,
                                     target_relation_vars: List[str]):
        # Forward: generate chain of new atom tests
        # Backward: generated possible aggregator combinations
        # print("generating attr's from ", current_var_names, current_atom_tests, sep="\n")
        if len(current_atom_tests) < self.longest_atom_test_chain:
            # reset everything
            current_atom_tests = []
            current_var_names = current_var_names[:1]

        if len(current_var_names) - 1 != len(current_atom_tests):
            raise WrongValueException(
                "Lengths do not differ by one::\n{}\n{}".format(
                    current_var_names, current_atom_tests))
        var_names_up_to_here = {}
        current_atom_tests_modified = []

        for r, vs, a in current_atom_tests:
            current_atom_tests_modified.append(
                (r.get_name(), list(zip(vs, r.get_types()))))
        for start_index in range(1, 1 + len(current_var_names)):
            var_names_up_to_here = union_of_two_dicts(
                var_names_up_to_here, current_var_names[start_index - 1])
            test_chain = current_atom_tests_modified[:start_index - 1]
            for new_tests_chain in self.generate_possible_attributes_helper_all_steps(
                    1, var_names_up_to_here):
                created_chain = test_chain + new_tests_chain
                if is_relation_chain_valid(created_chain):
                    fresh_v, fresh_i = DecisionTree.fresh_in_last_relation(
                        created_chain, target_relation_vars)
                    aggregator_chains = self.generate_possible_aggregator_chains(
                        len(created_chain), fresh_v, fresh_i)
                    yield start_index, created_chain, aggregator_chains, (
                        set(fresh_v), fresh_i)

    @staticmethod
    def fresh_in_last_relation(tests_chain, target_relation_vars):
        # compute the last fresh variable(s)
        last_fresh = []
        last_fresh_indices = []
        initial_var_names_set = set(target_relation_vars)
        non_fresh = set()
        for test in tests_chain[:-1]:
            _, var_names_types = test
            non_fresh |= set(var_names_types)
        relation_name, var_names_types = tests_chain[-1]
        for i, var_name_type in enumerate(var_names_types):
            var_name, var_type = var_name_type
            p0 = var_name[0] != "C"
            p1 = var_name not in initial_var_names_set
            p2 = var_name_type not in non_fresh
            p3 = var_type != Settings.atom_test_type_constant
            if p0 and p1 and p2 and p3:
                last_fresh.append(var_name_type)
                last_fresh_indices.append(i)
        return last_fresh, last_fresh_indices

    def generate_possible_aggregator_chains(self, tests_chain_len,
                                            last_fresh_variables,
                                            last_fresh_indices):
        def last_step_helper(fresh: int, current_type: str):
            if fresh == 0:
                # In this case, the aggregate should return 0 from [0] and 1 from [1] --> sum suffices
                options = []
                is_complex = False
            else:
                if current_type == TYPE_TUPLE:
                    options = COMPLEX_ENOUGH_AGGREGATORS
                    is_complex = True
                elif current_type == TYPE_NOMINAL:
                    options = NOMINAL_AGGREGATORS
                    is_complex = False
                elif current_type == TYPE_NUMERIC:
                    options = NUMERIC_AGGREGATORS
                    is_complex = False
                elif current_type == TYPE_TYPE:
                    # some user defined type like Person, Atom etc.
                    options = NOMINAL_AGGREGATORS
                    is_complex = False
                else:
                    raise WrongValueException(
                        "Unknown type: {}".format(current_type))
            chosen = []
            if is_complex and PROJECT.name in self.allowed_aggregators:
                for i, var_index in enumerate(last_fresh_indices):
                    chosen += [
                        PROJECTIONS[var_index](o)
                        for o in last_step_helper(1, last_super_types[i])
                        if o.name != COUNT.name
                    ]
            chosen += [
                a for a in options if a.name in self.allowed_aggregators
                and a.name != PROJECT.name
            ]
            yield from chosen

        def generator_chains_helper(current_options_gen, current_super_type,
                                    current_types, position):
            current_options = list(current_options_gen)
            if force_use_sum[0]:
                if len(current_options) == 0:
                    current_options = {SUM}
                else:
                    force_use_sum[0] = False
            for o in current_options:
                next_super_type, next_types = get_out_type(
                    current_super_type, current_types, o)
                if position == 0:
                    if o not in forbidden_first:
                        yield [o], next_types[0]
                else:
                    i = None
                    o_real = o.aggregator if o.is_projection else o
                    for i, g in enumerate(groups):
                        if o_real in g:
                            break
                    next_options = is_followed_by[next_super_type][i]
                    for left_part, left_type in generator_chains_helper(
                            next_options, next_super_type, next_types,
                            position - 1):
                        yield left_part + [o], left_type

        def get_super_type(k, t):
            """
            :param k: number of variables that are considered
            :param t: ignored if k != 1, variable type as given in relation definition in the settings.
            :return: Type of (the group of) the fresh variables
            """
            # behaves as numeric if no fresh
            # or precisely one which is numeric
            if k == 0:
                return TYPE_NUMERIC
            elif k > 1:
                return TYPE_TUPLE
            else:
                if t.startswith(Relation.constant_numeric):
                    return TYPE_NUMERIC
                elif t.startswith(Relation.constant_nominal):
                    return TYPE_NOMINAL
                else:
                    # user defined
                    return TYPE_TYPE

        def get_out_type(current_super_type, current_types,
                         a: Union[Aggregator, Project]):
            t = a.output_type
            if current_super_type == TYPE_TUPLE:
                # counting or projecting
                if t == TYPE_NUMERIC:
                    # counting, numeric projections
                    return TYPE_NUMERIC, [
                        TYPE_NUMERIC
                    ]  # new type can be also simply numeric
                elif t == TYPE_SAME_AS_INPUT:
                    # nominal projection
                    if not a.is_projection:
                        raise ValueError("Should be projection")
                    new_type = current_types[absolute_to_relative_positions[
                        a.component]]
                    new_super_type = get_super_type(1, new_type)
                    return new_super_type, [new_type]
                else:
                    raise WrongValueException(
                        "Wrong output type: {}".format(t))
            else:
                if t == TYPE_SAME_AS_INPUT:
                    return current_super_type, current_types
                elif t == TYPE_NUMERIC:
                    return TYPE_NUMERIC, [
                        TYPE_NUMERIC
                    ]  # new type can be also simply numeric
                else:
                    raise WrongValueException(
                        "Wrong output type: {}".format(t))

        group0 = {MIN, MAX, MEAN, SUM}
        group1 = {COUNT, COUNT_UNIQUE}
        group2 = {MODE}
        group3 = {FLATTEN, FLATTEN_UNIQUE}
        groups = [[a for a in g if a.get_name() in self.allowed_aggregators]
                  for g in [group0, group1, group2, group3]]
        forbidden_first = {FLATTEN_UNIQUE, FLATTEN}
        base_followed_dict = {
            0: groups[0],
            1: groups[0],
            2: groups[1] + groups[2] + groups[3]
        }
        is_followed_by_numeric = {x: y for x, y in base_followed_dict.items()}
        is_followed_by_nominal = {x: y for x, y in base_followed_dict.items()}
        is_followed_by_tuple = {x: y for x, y in base_followed_dict.items()}
        is_followed_by_type = {x: y for x, y in base_followed_dict.items()}
        # update with group3 followers
        is_followed_by_numeric.update({3: groups[0] + groups[1] + groups[3]})
        is_followed_by_nominal.update({3: groups[1] + groups[2] + groups[3]})
        is_followed_by_tuple.update({3: groups[1] + groups[3]})
        is_followed_by_type.update({3: groups[1] + groups[2] + groups[3]})
        is_followed_by = {
            TYPE_NUMERIC: is_followed_by_numeric,
            TYPE_NOMINAL: is_followed_by_nominal,
            TYPE_TUPLE: is_followed_by_tuple,
            TYPE_TYPE: is_followed_by_type
        }
        last_types = [value_type for _, value_type in last_fresh_variables]
        last_super_types = [
            get_super_type(1, value_type)
            for _, value_type in last_fresh_variables
        ]
        nb_fresh = len(set(last_fresh_variables))
        last_fresh_type = None if nb_fresh != 1 else last_fresh_variables[0][1]
        last_super_type = get_super_type(nb_fresh, last_fresh_type)
        force_use_sum = [nb_fresh == 0
                         ]  # to prevent initialisation in helping functions
        absolute_to_relative_positions = {
            a: r
            for r, a in enumerate(last_fresh_indices)
        }
        yield from generator_chains_helper(
            last_step_helper(nb_fresh, last_super_type), last_super_type,
            last_types, tests_chain_len - 1)

    def generate_possible_attributes_helper_all_steps(
            self, depth, var_counts_up_to_here: Dict[str, Set[str]]):
        for head in self.generate_possible_attributes_helper_one_step(
                var_counts_up_to_here):
            yield [head]
            if depth < self.max_number_atom_tests:
                # update counts
                _, var_names_types = head
                fresh_dict = {}  # type: Dict[str, Set[str]]
                for var_name, var_type in var_names_types:
                    if var_type not in fresh_dict:
                        fresh_dict[var_type] = set()
                    fresh_dict[var_type].add(var_name)
                new_counts = union_of_two_dicts(var_counts_up_to_here,
                                                fresh_dict)
                for tail in self.generate_possible_attributes_helper_all_steps(
                        depth + 1, new_counts):
                    yield [head] + tail

    def generate_possible_attributes_helper_one_step(self,
                                                     var_counts_up_to_here):
        for rel_spec in sorted(self.allowed_atom_tests):
            counts = self.allowed_atom_tests[rel_spec]
            relation_name, var_specifications, var_types = rel_spec
            arity = len(var_specifications)
            o_types = sorted(counts.keys())
            for needed_keys, configuration in generate_variable_configurations(
                    var_counts_up_to_here, counts, o_types):
                assert len(configuration) == len(counts)  # one for each type
                variable_names = [None for _ in range(arity)]
                none_counter = arity
                for type_config, o_type in zip(configuration, o_types):
                    (ind_old, ind_new), (names_old,
                                         relative_counts_new) = type_config
                    # place old
                    for i_old, n_old in zip(ind_old, names_old):
                        assert variable_names[i_old] is None
                        none_counter -= 1
                        variable_names[i_old] = n_old
                    # place new
                    temp_new_var_names = self.generate_temp_var_names(
                        "Y", relative_counts_new)
                    for i_new, n_new in zip(ind_new, temp_new_var_names):
                        assert variable_names[i_new] is None
                        none_counter -= 1
                        variable_names[i_new] = n_new
                    # place constants
                    for i_const in counts[o_type][2]:  # TODO: hard-coded 2 ...
                        assert variable_names[i_const] is None
                        none_counter -= 1
                        variable_names[i_const] = self.generate_temp_var_name(
                            "C")
                assert none_counter == 0
                var_names_types = list(zip(variable_names, var_types))
                yield relation_name, var_names_types

    def create_example_and_chains(self, relations, current_vars):
        def const_value_generator(c_name):
            r_name, position = constants[c_name]
            yield from self.descriptive_data[r_name].get_all_values(position)

        def num_const_values():
            product = 1
            for c_name in constant_var_names:
                r_name, position = constants[c_name]
                product *= self.descriptive_data[r_name].get_nb_all_values(
                    position)
            return product

        relation_chain = []
        example = {}
        constants = {}
        # current
        for d in current_vars:
            for vs in d.values():
                for v in vs:
                    n = v.get_name()
                    assert n not in example
                    example[n] = v
        # extended
        for relation_name, var_names in relations:
            for i, (n, t) in enumerate(var_names):
                if n not in example:
                    if n[0] == "C":
                        assert n not in example  # always fresh constant variables
                        example[n] = ConstantVariable(n, t, None)
                        constants[n] = (relation_name, i)
                    elif n[0] in "XY" and n not in example:
                        example[n] = VariableVariable(n, t, None)
                    elif n[0] not in "CXY":
                        raise WrongValueException(
                            "Wrong variable name {}".format(n))
            relation_chain.append((self.descriptive_data[relation_name],
                                   [n for n, _ in var_names]))

        constant_var_names = sorted(constants.keys())
        c_values = itertools.product(
            *[const_value_generator(c) for c in constant_var_names])
        return example, relation_chain, c_values, constant_var_names, num_const_values(
        )

    def predict(self, d: Datum, is_for_ensemble=False):
        example = {}  # type: Dict[str, Variable]
        for var, value in zip(self.target_relation_variables,
                              d.get_descriptive()):
            example[var.get_name()] = var
            var.set_value(value)
        current_node = self.root_node  # type: TreeNode
        while not current_node.is_leaf():
            # extend the example
            for name, var in current_node.get_split().get_fresh_variables(
            ).items():
                example[name] = var
            # get test value and send example to one of the children
            is_positive = current_node.get_split().evaluate(example)
            if is_positive:
                current_node = current_node.get_child(TreeNode.positive_branch)
            else:
                current_node = current_node.get_child(TreeNode.negative_branch)
        if is_for_ensemble:
            prediction = current_node.get_stats().get_prediction_for_ensemble()
        else:
            prediction = current_node.get_stats().get_prediction()
        for var in example.values():
            if var.can_vary():
                var.unset_value()
        return prediction

    def predict_all(self, ds: List[Datum], is_for_ensemble=False):
        return [self.predict(d, is_for_ensemble) for d in ds]

    def target_data_induction_preparation(self, target_data: List[Datum]):
        if self.class_weights is None:
            return None
        for d in target_data:
            d.set_weight(d.get_weight() * self.class_weights[d.get_target()])

    def reverse_target_data_induction_preparation(self,
                                                  target_data: List[Datum]):
        if self.class_weights is None:
            return None
        for d in target_data:
            d.set_weight(d.get_weight() / self.class_weights[d.get_target()])

    @staticmethod
    def populate_dict(target_data: List[Datum]):
        for datum in target_data:
            if datum.identifier not in TEST_VALUE_MEMO:
                TEST_VALUE_MEMO[datum.identifier] = {}

    @staticmethod
    def test_values_memo_keys(example: Dict[str, Variable],
                              relation_chain: List[Tuple[Relation, List[str]]],
                              aggregation_chains: List[Tuple[Aggregator]]):
        """

        :param example:
        :param relation_chain:
        :param aggregation_chains:
        :return: A 3-tuple: (relations_as_key, aggregators_as_keys, known_unknown_separated) where
        - relations_as_key: (relation name, known_unknown, values) where
            - relation name: str
            - known_unknown: [is value known for variable1, ...]
            - values:
                - if variable is set to some value, then this equals its value
                - elif variable is input variable, i.e., starts with 'X': name of the variable
                - else: an integer
        - aggregators_as_keys: tuple of tuples of aggregator names
        - known_unknown_separated: indices of the variables with (un)known values
        """
        parts = []
        known_unknown_separated = []
        counters = {}
        max_overall = -1
        for r_vs in relation_chain:
            r, vs = r_vs
            known_unknown = []
            known_unknown_separated_part = [[], []]  # type: List[List[int]]
            out_values = []
            max_part = -1
            for i, v in enumerate(vs):
                u = example[v].is_unset(
                )  # target variables should not be set here yet
                known_unknown.append(not u)
                if u:
                    if v[0] == 'X':
                        out_values.append(v)
                        known_unknown_separated_part[0].append(
                            i)  # but are considered known
                    else:
                        if v not in counters:
                            counters[v] = len(counters)
                        value = counters[v]
                        out_values.append(value)
                        if value <= max_overall:
                            known_unknown_separated_part[0].append(
                                i)  # seen in the previous parts
                        else:
                            known_unknown_separated_part[1].append(
                                i)  # unknown until this part
                        max_part = max(max_part, value)
                else:
                    # assert v[0] != "X"
                    out_values.append(example[v].get_value())
                    known_unknown_separated_part[0].append(i)  # known
            parts.append((r.name, tuple(known_unknown), tuple(out_values)))
            max_overall = max(max_part, max_overall)
            known_unknown_separated.append(known_unknown_separated_part)
        relations_as_key = tuple(parts)
        aggregators_as_keys = [
            tuple(a.name for a in a_s) for a_s in aggregation_chains
        ]
        return relations_as_key, aggregators_as_keys, known_unknown_separated


def is_relation_chain_valid(chain: List[Tuple[str, List[Tuple[str, str]]]]):
    """
    We check whether
    0) At least one of the fresh (Y) variables in the relation r_i is used in r_{i + 1}
    1) 2nd, 3rd, ... relation contains at least one variable from the previous relation.
    2) no two elements are equal
    :param chain: [(rel1, var_name_types1), ...], var_name_types1 = [('Y2', 'Person'), ...]
    :return:
    """
    # 0)
    for i in range(len(chain) - 1):
        link_ok = True
        has_fresh = False
        for name, _1 in chain[i][1]:
            if name[0] == "Y":
                has_fresh = True
                link_ok = False
                for next_name, _2 in chain[i + 1][1]:
                    if name == next_name:
                        link_ok = True
                        break
            if has_fresh and link_ok:
                break
        if not link_ok:
            return False
    # 1)
    for i in range(1, len(chain)):
        link_ok = False
        for name, _1 in chain[i][1]:
            for prev_name, _2 in chain[i - 1][1]:
                if prev_name == name:
                    link_ok = True
                    break
            if link_ok:
                break
        if not link_ok:
            return False
    # 2)
    chain_copy = chain[::]
    chain_copy.sort()
    for i in range(1, len(chain)):
        if chain_copy[i] == chain_copy[i - 1]:
            return False
    return True


def generate_variable_configurations(present, needed, needed_keys):
    def generate_config_one_type(p, n):
        old, new, _ = n
        for additional_old, brand_new in subsets_of_list(new):
            true_old = old + additional_old
            true_new = brand_new  # new + brand_new?
            old_configs = generalized_counting(
                sorted(p), len(true_old))  # [['X1', 'X2', 'X1'], ...]
            new_configs = canonic_sequences_of_new_variables(
                len(true_new))  # [[1, 3, 2, 1], ...]
            for combo in itertools.product(old_configs, new_configs):
                yield (true_old, true_new), combo

    # extend present with empty lists
    for t in needed_keys:
        if t not in present:
            present[t] = set()
    pairs = [(present[t], needed[t]) for t in needed_keys]
    # cartesian product over the types
    for combined in itertools.product(
            *[generate_config_one_type(p, n) for p, n in pairs]):
        yield needed_keys, combined


def random_subsets(a_list, nb_subsets, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    for _ in range(nb_subsets):
        left_right = [[], []]
        for x in a_list:
            left_right[random.random() > 0.5].append(x)
        yield left_right


def create_new_variable(v_name, v_type, value=None):
    if v_name[0] == "C":
        return ConstantVariable(v_name, v_type, value)
    elif v_name[0] in "XY":
        return VariableVariable(v_name, v_type, value)
    elif v_name[0] not in "CXY":
        raise WrongValueException("Wrong variable name {}".format(v_name))


def create_constant_tree(heuristic: Heuristic, data: Dataset):
    t = DecisionTree(heuristic=heuristic, max_number_internal_nodes=0)
    t.build(data)
    return t
