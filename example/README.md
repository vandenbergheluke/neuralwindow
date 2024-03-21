# Examples

## Description

Example models using neuralwindow. Build them from the project directory via the commands

```bash
mkdir build
cd build
cmake .. && make
./example/<model>
```

## Deep Feed Forward Neural Network - MNIST Classifier

<p align="center">
    <img src="img/mnist_accuracy_train.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/mnist_cost_train.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/mnist_accuracy_valid.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/mnist_cost_valid.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/mnist.png">
</p>

## Deep Convolutional Generative Adversarial Neural Network - MNIST

Generate image with trained GAN. Image saved in `build/img/demo.png`.

```bash
DEMO=1 ./example/deep_convolutional_gan ../example/models/generator.bin
```

<p align="center">
    <img src="img/demo.png", height=300, width=300>
</p>

<p align="center">
    <img src="img/generator_cost.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/discriminator_cost.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/total_cost.png", height=300, width=600>
</p>

## Transformer - Simpsons Dataset

Transfomer trained on [Simpsons Dataset](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset)

Generate simpsons style quotes with the trained transfomer. Optional custom prompt can be provided. Default prompt is '\n'.

```bash
DEMO=1 ./example/transformer ../example/models/transformer.bin '<prompt>'
```

```
$ DEMO=1 ./example/transformer ../example/models/transformer.bin 'Homer'
Output: is box taste with his biket. Check two!
Not just there.
Bombo, are there you going?
Come on, Homer.
Allson owns asked miser. Just me a whewsillant won't sit bomboling here, and I do.
Enough, Homer!
Come on with the hight her in a scottown for a lot fire walk?
Right, Apu. Make full of that sching on Homey news, wouldn't you?
Apu ain't a postainly all that horse to telf. But here, I see  they've worked watching me "Special Cool."
```
<p align="center">
    <img src="img/transformer_cost_train.png", height=300, width=600>
</p>
<p align="center">
    <img src="img/transformer_cost_valid.png", height=300, width=600>
</p>