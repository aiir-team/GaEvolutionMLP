# GaEvolutionMLP
- Using Genetic Algorithm to find the optimal network parameters of Multi-Layer Perceptron for classification tasks

On the easy MNIST dataset, we are able to quickly find a network that reaches > 98% accuracy. 
On the more challenging CIFAR10 dataset, we get to 56% after 10 generations (with population 20).


## To run

To run the brute force algorithm:

```python3 brute.py```

To run the genetic algorithm:

```python3 main.py```

You can set your network parameter choices by editing each of those files first. You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `dataset` to either `mnist` or `cifar10`.

## Credits

+ The genetic algorithm code is based on the code from this excellent blog post: https://lethain.com/genetic
-algorithms-cool-name-damn-simple/

+ The small version in: https://github.com/harvitronix/neural-network-genetic-algorithm

+ The big version in: [Jan Liphardt's implementation, DeepEvolve](https://github.com/jliphard/DeepEvolve).


## Contributing

Have an optimization, idea, suggestion, bug report? Pull requests greatly appreciated!

## License

MIT
