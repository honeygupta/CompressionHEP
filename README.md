# Deep-compression for HEP data
This repository contains the fastai(Pytorch) implementation for Deep-compression for HEP data and includes the scripts to train and test different variants of the network. 
It also contains scripts to plot the errors and analyze the results. 

The code is developed by [Honey Gupta](https://github.com/honeygupta) and is derived from the works of [Erik Wallin](https://github.com/Skelpdar) 
and [Eric Wulff](https://github.com/erwulff). Theoretical explanations and other details can be found in Eric Wulff's [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751).
 

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
1a. 4D_3D_200_no_norm.ipynb
* Base model trained on non-normalized data

***
1b. 4D_3D_200_standard_norm.ipynb
   * Base model trained on standard-normalized data
***
1c. 4D_3D_200_custom_norm.ipynb
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

* High bias in the non-normalised model. pt has low error whereas eta very high reconstruction error.
* Standard normalization has highest error for most of the parameters among the three models (based on the variance of the errors)
* Custom normalization has better performance for most of the parameters as compared to the others (based on variance and MSE)

Since custom norm data produced the lowest mean squared error on the test-set and is able to capture correlations among variables in a better way, I chose custom normalization for all further experiments.
 
 
 #### Model comparison
| Model         |     m    | p_t        | \phi       | \eta       | MSE on the test-set |
|---------------|:--------:|------------|------------|------------|---------------------|
| Tanh, no NN   | 0.010748 | -0.0001039 | 0.0007383  | 0.002638   | 0.0007314           |
| LeakyReLU, BN | 0.004853 | -0.001126  | 0.008089   | -0.02475   | 0.0005750           |
| ELU, BN       | 0.005528 | -0.0007568 | -0.003144  | -0.0002156 | 0.0005754           |

* LeakyReLU model has moderate performance (based on the variance of relative error and MSE on the test-set).
* Tanh and ELU have comparable performance. Tanh has lower variance and mean for the relative error but ELU has lower MSE.
Hence ELU  model can be said to be the better among by considering both the relative error and MSE, since there is not much difference between ReLU and ELU’s MSE for the test-set.


#### Execution time and memory allocation comparison
| Model         | Model initialization time (s) | Model load time (s) | Encoding time (s) | Decoding time (s) | Encoding memory alloc (MB) | Decoding memory alloc (MB) |
|---------------|:-----------------------------:|---------------------|-------------------|-------------------|----------------------------|--------------------|
| Tanh, no NN   | 2.5174                        | 0.05932             | 0.03934           | 0.02113           | 0.0024                     | 0.1780             |
| LeakyReLU, BN | 2.5903                        | 0.08180             | 0.12613           | 0.10033           | 0.0076                     | 0.2240             |
| ELU, BN       | 2.4373                        | 0.07940             | 0.16376           | 0.15237           | 0.0064                     | 0.1945             |

* ELU has exponential component,  hence it’s runtime is expected to be higher
* Tanh also has exponential component, but the batchnorm layer in other two models increases execution time.
Overall, the base tanh model seems to be the best in terms of execution time and memory allocation. 




 
