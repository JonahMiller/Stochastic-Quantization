# L46-Project

Investigation into [Stochastic Quantization](https://arxiv.org/pdf/1708.01001.pdf) (SQ). This for the L46 Module in the Computer Science Part III course at the University of Cambridge.

## Stochastic Quantization in PyTorch

This project recreates the stochastic algorithm in PyTorch, and can run on low-bit DNNs: BWN, TWN or TTQ.

There are two model setups: VGG-9, ResNet-20. However, general VGG and ResNet models can be setup in the models.py. There are differences between the models I use and those in the paper.

The quantization model adaptions are in the util.py file, and the main.py file runs it. 

There are multiple quantization approaches that can be called under the arguments seen in the *args.quant* in the main.py file. 

`python --quant sq_bwn_default_layer --model vgg9 `
or 
`python --quant sq_twn_default_layer --model vgg9 `

Runs the program with the SQ algorithm defined by the paper.

## Additional adaptations

I created additional adaptions to run the algorithm in a non-selection manner (ie. selects which layers do not get quantized), this is furthered by an approach which quantizes by filters/elements in a layer, rather than layers in the model. This also has different mechanisms of gaining the probabilities of quantization, which can be seen under *args.e_type* and *args.prob_type*.

I run a series of tests  with these settings and these can be seen in the [test.py](./testing/test.py) file. All models used in my project from here are in the [trained_models](./trained_models/) folder. The accuracy and other details are recorded in [final.txt](./txt_results/final.txt).

## ELQ

I also write and then add stochasticity to an ELQ based algorithm. The code can be found primarily in the [inq](./inq/) folder, and it is run from main2.py.

Again a series of tests have been run demonstrating usage in the [test2.py](./testing/test2.py) file. All models used in my project from here are in the [trained_models2](./trained_models2/) folder. The accuracy and other details are recorded in [final2.txt](./txt_results/final2.txt).

## Additional comments

Any code that has either been based on or copied from another project is commented at the top of that page.
