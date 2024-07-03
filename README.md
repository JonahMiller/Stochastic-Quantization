# An Investigation into Stochastic Quantization

An attempt to reproduce and improve the results from [Stochastic Quantization](https://arxiv.org/pdf/1708.01001.pdf) (SQ) using PyTorch. The initial repo from the paper is written in Caffe and can be found [here](https://github.com/dongyp13/Stochastic-Quantization/tree/master).

## Stochastic Quantization in PyTorch

This project recreates the stochastic algorithm in PyTorch, and can run on low-bit DNNs: BWN, TWN or TTQ.

There are two model setups: VGG-9 and ResNet-20. However, general VGG and ResNet models can be setup in the [model.py](./model.py). There are differences between the models I use and those in the paper.

The quantization model adaptions are in the util.py file, and the main.py file runs it. 

There are multiple quantization approaches that can be called under the arguments seen in the *args.quant* in the main.py file. 

`python --quant sq_bwn_default_layer --model vgg9 `
or 
`python --quant sq_twn_default_layer --model vgg9 `

Runs the program with the SQ algorithm defined by the paper.

## Additional adaptations

I created additional adaptions to run the algorithm in a non-selection manner (ie. selects which layers do *not* get quantized), this is furthered by an approach which quantizes by filters/elements in a layer, rather than layers in the model. This also has different mechanisms of gaining the probabilities of quantization, which can be seen under *args.e_type* and *args.prob_type*.

I run a series of tests  with these settings and these can be seen in the [test.py](./testing/test.py) file.

## ELQ

I also write and then add stochasticity to an ELQ based algorithm. The code can be found primarily in the [inq](./inq/) folder, and it is run from main2.py.

Again a series of tests have been run demonstrating usage in the [test2.py](./testing/test2.py) file.

## Additional comments

This project was initially done to the L46 module for the Part III Computer Science course at the University of Cambridge. The final report is in the [report.pdf].
