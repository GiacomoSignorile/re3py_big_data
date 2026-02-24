# RE3PY - relational tree learners in python!
Welcome to RE3PY's official repo. To install the library, please try 
(latest):
```
pip install -r requirements.txt

python setupy.py install
```
For additional visualizations, please also install py3plex library via `pip install py3plex` call.
# Experiments (data reduction)
For the complete guide to the `experiment/` pipeline (preprocessing, fold mapping, model runs, analysis), see:

- `experiment/README.md`

Quick start:

```bash
cd experiment
python data_scarcity_preprocessing.py --dataset basket --seed 2864
python fold_mapper.py --dataset basket
python run_experiment_bagging_full_test.py --dataset basket --config-name agg_all --n-jobs 2
```
# Testing
To execute the series of tests which determine proper installation, execute:
```
python -m pytest tests
```
![gif-reel](images/reel.gif)

# Full replicability
To fully replicate the working environment, we offer a singularity container.
Go to ./container folder and run
```
bash make.sh
```
Note that you will need [singularity](https://github.com/hpcng/singularity) installed for this to work.
An image called ranking.sif will be created in ./container folder. You can run arbitrary scripts that include rrank library by simply doing:
```
singularity exec ranking.sif python somescript.py --some_parameter --some_other_parameter 
```

# Usage examples
The series of core examples is available in the ./examples folder.


# A quick word about the method

## Relations
In the code and in the input files, relations are thought of as sets whose elements are given exhaustively. Every such set contains tuples of objects of given types.
For example, if `Ana likes potato`, `Ana likes beef`, `Bob likes beef`, `Jacques likes omlette`, etc., this can be encoded as the relation

`likes(Person, Food) = {(Ana, potato), (Ana, beef), (Bob, beef), (Jacques, omlette), ...}`.

The relation does not have to be binary. Any other arity (greater or equal to 1) is possible. For example, the ternary relation `loveTriangle(Person, Person, Person)` is given as  

`loveTriangle(Person, Person, Person) = {(Ana, Bob, Charles), (Denis, Dolores, Deborah), ...}`.

## Feature construction
The method creates (ensembles of) trees where the internal nodes split the data into two groups according to the test `f(X) in S`, where `X` is the input example, `f` is a feature and `S` is some set of values. For example, the feature

`f(P1) = mean_P2 count_P3 friend(P1, P2) and friend(P2, P3)`

gives the average numer of friends that a friend of person `P1` has. A corresponding test would be `f(P1) in [10, infinity)` (i.e., `f(P1) >= 10`).

In general, the features are of the form

`f(X) = agg1_X1 agg2_X2 ... aggN_Xn r1(A1, A2, ...) and r2(B1, B2, ...) and ...`

where

- `r1`, `r2`, ... are relations
- `X1`, `X2`, `A1`, `A2`, `B1`, ... are variables (not necessarily all differnet - in the example above, `X1 = A2 = B1`)
- `X` is the input example (it may appear as one of the variables `Ai` or `Bi`, but not as any of the variables `Xi`).

The set `S` is of form 

- `[x, infinity)` or `(-infinity, x]` (for some number x) if the aggregate `agg1` returns numeric values (see below), or
- `S = {x1, x2, ..., xM}` where `xi` are some nominal values from the domain of `X1`.


# Syntax

In order to grow a tree (or an ensemble of trees), we need

- a file with descriptive relations,
- a file with target relation,
- a settings file.

The following sections describe each of them.

## Descriptive- and Target-relation Files
The relations are divided into two groups:

- **Descriptive relations** are the relations that can be used in the tree splits. They can be unary, binary, ternary, etc.
- **Target relation** is the relation whose elements need to be predicted. For technical reasons, the target relation **cannot be unary**.

### Descriptive-relations file
Descriptive-relations file contains the elements of all the descriptive relations, one element of a relation per line:

```
likes(Ana, potato)
likes(Ana, beef)
likes(Bob, beef)
likes(Jacques, omlette)
...
loveTriangle(Ana, Bob, Charles)
loveTriangle(Denis, Dolores, Deborah)
...
```

The spaces are optional and the order of the facts is not important.
To ignore a line, use `//` (the part of the line after `//` is ignored).

### Target-relation file

This file contains only the elements of the target relation (the relation whose elements need to be predicted). The same syntax as above applies.
For example, if we are predicting the whether a person is vegetarian or not, the target relation should be given as

```
vegetarian(Ana, no)
vegetarian(Bob, no)
vegetarian(Jacques, yes)
vegetarian(Charles, no)
...
```

For a given datum, e.g., `vegetarian(Person, nominal)`, a learner uses only the `Person` part and returns a value of the type `nominal`.
The values that we are predicting (above, these are the `yes` and `no` values) should always be the last component in the tuples.

Note that `vegetarian` is actually a unary relation. The trick with the additinal component is only necessary to make other problems (including multi-class or regression) simpler.
For example, we can predict relations such as

- `friends(Person, Person, nominal) = {(Ana, Bob, yes), (Ana, Charles, yes), (Ana, Denis, no), ...}` where (given a pair of `Person`s) `yes` or `no` is predicted;
- `age(Person, numeric) = {(Ana, 19), (Bob, 18), (Denis, 45), ...}` where (given a `Person`) the numeric value is predicted;
- ...


## Settings File
The settings file has three important sections:

- relations
- aggregates
- atom tests

The syntax is the following:

```
[Relations]
targetRelation(type01, type02, ...)
descriptiveRelation1(type11, type12, ...)
descriptiveRelation2(type21, type22, ...)
...
[Aggregates]
aggregate1
aggregate2
...
[AtomTests]
descriptiveRelation1(new/old/c,new/old/c,...)
...
```
There can be empty lines. The part of the line after `//` is ignored, i.e., `//` works as a comment.

### Relations
After the section name in the square brackets, every line gives a relation name, together with the types of the components of the tuples in this relation.
The first listed relation is the target relation. Then, descriptive relations follow in any order. For example,

```
[Relations]
vegetarian(Person, nominal)
likes(Person, Person)
loveTriangle(Person, Person)
...
```

Type names must follow any of the following rules:

- The type starts with `nominal` (e.g., `nominal`, `nominalColor`, `nominal2`, etc.). This tells the program that nominal aggregates (see below) can be used for such variables.
- The type starts with `numeric` (e.g., `numeric`, `numericAge`, `numeric2`, etc.). This tells the program that numeric aggregates (see below) can be used for such variables.
- The type starts with a capital letter (e.g., `Person`, `MovieID`, etc.). These are "user-defined" types. Only general aggregates (see below) such as count can be used for such variables.

The type of the target should be nominal or numeric.

### Aggregates
After the section name in the square brackets, every line gives a aggregate name. The *true aggregates* work on the lists of values and return a single value. Three of them (flatten, flatten unique and projection) are a bit special.

The supported *true aggregates* are:

- count, count unique: count the number of (unique) appearances in the list, e.g., `count([A, B, C, A]) = 4`, `countUnique([A, B, C, A]) = 3`. Can be used on lists of any type of elements.
- min, max, mean, sum: give min/max/mean or sum of the list, e.g., `sum([1, 2, 3]) = 6`. Can be used on lists with numeric elements.
- mode: Gives the mode of the list, e.g., `mode([A, B, C, A]) = A`. Can be used on lists with nominal and user-defined elements. Ties are resolved alphabetically, e.g., `mode([B, A, B, A]) = A`).

The special aggregates are:
- flatten, flatten unique: When searching for related objects, a list of lists appear. In the example above (friends of friends), on the first step, we find all the friends of a given `Person` `X` (represented as a list of `Person`s, e.g., `[Ana, Bob]`), and on the second step, we find friends of friends (represented as a list of lists of `Person`s, e.g., `[[Bob, Jacques, Bill], [Jacques, Bill, John]]` - a list for Ana and a list for Bob). It may be beneficial to simply flatten this into `[Bob, Jacques, Bill, Jacques, Bill, John]` (or additionally remove duplicates) and apply any of the *true aggregates* in the next steps. Flatten and flatten unique cannot be used in the last step of aggregation (i.e., as the aggregate `agg1` (following the example above)
- projection: When searching for related objects, more than one variable might be introduced. To aggregate only one of them, projection (toghether with any of the *true aggregates* is used).


### Atom tests

Every feature

`f(X) = agg1_X1 agg2_X2 ... aggN_Xn r1(A1, A2, ...) and r2(B1, B2, ...) and ...`

consists of atom parts `r1(A1, A2, ...)`, `r2(B1, B2, ...)`, ... When adding a new atom part to the feature, we can pose some additional constraints on the those parts.
For example, it does not make sense to construct features such as

`f(P1) = mean_P2 count_P4 friend(P2, P3) and friend(P4, P5)`

so we may want to ensure that

- `P1 = P2` or `P1 = P3`, and
- any of the variables `P4` and `P5` equals any of the variables `P2` and `P3`. This can be done with the following lines:

```
[AtomTests]
friend(new, old)
friend(old, new)
...
```
that specify (respectively) that 

- the first variable can be new (but not necessarily) and the second one must be known before, when the relation `friend` is used, or
- vice-versa: the second variable can be new, and the first one must be known before, when the relation `friend` is used.

In addition to the keywords `new` and `old`, the `c` keyword can be used. In that case, instead of the variable, a constant from its variable's domain is used.


### Other Parameters
are passed directly to the learner's constructor (e.g., depth of a tree, maximal number of atom tests in a split etc.)


# Tabular data example
Dataset in the stadard form

| x1  | ... | xN |  y   |
| ----- | ---- | ---- | ----  |
| M     |  ... | 2.2 |  red   |
| F     |  ... | 3.2 |  black |
|  ...  | ... | ...  | ...    |

can be converted into descriptive relations

```
x1(example1, M)
...
xN(example1, 2.2)
x1(example2, F)
...
xN(example2, 3.2)
...
```

and target relation

```
y(example1, red)
y(example2, black)
...
```

If we then specify settings file

```
[Relations]
y(Example, nominal1)`
x1(Example, nominal2)
...
xN(Example, numericN)

[Aggregates]
mode
sum

[AtomTests]
x1(old, new)
x2(old, new)
...
xN(old, new)
```

(we only need to specify one aggregate for for nominal and one for numeric attribtues - does not matter which since all the relations are -to-one),
and set the number of atom tests in an internal node to 1, the resulting relational decision tree should be the same as the standard _non-relational_ decision tree for this dataset.

