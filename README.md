# Deep-compression for HEP data
This repository contains the fastai(Pytorch) implementation for Deep-compression for HEP data and includes the scripts to train and test different variants of the network. 
It also contains scripts to plot the errors and analyze the results. 

The code is developed by [Honey Gupta](https://github.com/honeygupta) and is derived from the works of [Erik Wallin](https://github.com/Skelpdar) 
and [Eric Wulff](https://github.com/erwulff). Theoretical explanations and other development details can be found in Eric Wulff's [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751).
 

## Getting Started
### Package requirements
 * The following packages are required to run the codes. Note: the version of the packages were the ones I used and are suggestive, not mandatory.
    * pandas = 1.0.1
    * torch = 1.4.0
    * fastai = 1.0.60
    * corner = 2.0.1
    * seaborn = 0.10.0  
    * jupyter notebook 

The models were trained on a Nvidia RTX 2080Ti GPU and evaluated on CPU. The scripts, by default, run on a GPU and might need small modifications to run a CPU.

### Repository structure
The folder "CompressionHEP" contains jupyter notebooks for training, testing and plotting the results for different variants of the model and the folder "datasets" contains the input and processed datasets. The folder/file descriptions are as follows:

* models - contains the models 
* utils - contains helper scripts 
* plots - contains the plots generated

***

0 . 4D_data_normalization.ipynb
* Script for generating normalized datasets with different normalization techniques: No norm, standard and custom normalization

***
1a. Pre-trained_performance_fastai_AE_3D_200_no1cycle.ipynb*
* Script for loading pretrained model from [HEPAutoencoders](https://github.com/Skelpdar/HEPAutoencoders) and testing on the given test-set.
 
***
1b. 4D_3D_200_no_norm.ipynb
* Base model trained on non-normalized data

***
 1c. 4D_3D_200_standard_norm.ipynb
   * Base model trained on standard-normalized data
***
1d. 4D_3D_200_custom_norm.ipynb
  * Base model trained on custom-normalized data
***
2a. 4D_3D_200_ReLU_BN_custom_norm.ipynb
* 7 hidden-layer model with ReLU activation and batch-normalization layer
***
2b. 4D_3D_200_ELU_BN_custom_norm.ipynb
* 7 hidden-layer model with ELU activation and batch-normalization layer 
***
Execution_time_calculation.ipynb
* Contains the scripts to calculate the execution time of different parts of network like data loading, network initialization, encoding and decoding.
***

Memory_allocation_trace.ipynb
* Contains the scripts to calculate the memory allocated by different parts of network.

***
##### Note that instructions for execution and other details can be found inside each notebook.
### Steps:

#### Normalize the data
Run the scripts in 0. 4D_data_normalization.ipynb to normalize the given data and create the dataset. 

#### Run the codes
The scripts to train, test and plot errors for each model have all included in the jupyter notebooks. 
 
#### Memory and execution time calculation
The scripts Execution_time_calculation.ipynb and Memory_allocation_trace.ipynb can be used to calculate the execution time and memory trace for all the models. Refer the instruction in the notebook for further details.

### Summary
#### Normalization comparison
| Normalization type |     m     | p_t          | \phi      | \eta     | MSE on the test-set |
|--------------------|:---------:|--------------|-----------|----------|---------------------|
| None               | -0.005725 | -0.000001969 | -0.004830 | -1.0107  | 0.5181              |
| Standard           | 0.001250  | 0.01578      | -0.01810  | 0.05978  | 0.01111             |
| Custom             | 0.010748  | -0.0001039   | 0.0007383 | 0.002638 | 0.0007314           |

  * We can observe high bias in the non-normalised model. p_t has very low error whereas \eta high very high reconstruction error.
  * Standard normalization has ~1e-2 error for all the parameters, whereas custom normalization has better performance for most of the parameters as compared to all 3. 
 
 #### Model comparison
 | Model         |     m    | p_t        | \phi       | \eta       | MSE on the test-set |
|---------------|:--------:|------------|------------|------------|---------------------|
| Pre-trained   | 3440.2*  | -0.02757   | -0.0898497 | 0.13439687 | 0.1864              |
| Tanh, no NN   | 0.010748 | -0.0001039 | 0.0007383  | 0.002638   | 0.0007314           |
| LeakyReLU, BN | 0.004853 | -0.001126  | 0.008089   | -0.02475   | 0.0005750           |
| ELU, BN       | 0.005528 | -0.0007568 | -0.003144  | -0.0002156 | 0.0005754           |
* Pretrained model has very high error, which indicates that there is a mismatch in the dataset distribution.
* LeakyReLU model has moderate performance. 
* Tanh and ELU have comparable performance. Both of them seem to be better for a set of variables, but tanh has error in order of 1e-2 for m, which can be considered as high comparatively. 
Hence ELU model can be said to be the best among these in terms of *mean relative reconstruction error*.

#### Execution time and memory allocation comparison
| Model         | Model initialization time (s) | Model load time (s) | Encoding time (s) | Decoding time (s) | Encoding memory alloc (MB) | Decoding memory alloc (MB) |
|---------------|:-----------------------------:|---------------------|-------------------|-------------------|----------------------------|----------------------------|
| Tanh, no NN   | 2.5174                        | 0.05932             | 0.03934           | 0.02113           | 0.0024                     | 0.1780                     |
| LeakyReLU, BN | 2.5903                        | 0.08180             | 0.12613           | 0.10033           | 0.0076                     | 0.2240                     |
| ELU, BN       | 2.4373                        | 0.07940             | 0.16376           | 0.15237           | 0.0064                     | 0.1945                     |

* ELU has exponential component,  hence it’s runtime is expected to be higher
* Tanh also has exponential component, but the batchnorm layer in other two models increases execution time.
* Overall, the base tanh model seems to be the best in terms of execution time and memory allocation. 




 