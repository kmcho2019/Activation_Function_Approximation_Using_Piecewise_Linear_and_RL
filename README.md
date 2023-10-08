# Activation Function Approximation Using Piecewise Linear and RL

## Description
A system for using RL to determine segment points for piecewise linear approximation of activation functions like sigmoid, SiLU(swish), GELU, et cetera. This project is part of the ISOCC2023 paper "Automated Generation of Piecewise Linear Approximations for Nonlinear Activation Functions via Curvature-based Reinforcement Learning".

## Technology Stack
- Python 3.11.2
- Stable Baselines3
- gymnasium

## Installation

``````bash
# Create virtual environment
python3.11 -m venv env

# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
``````

## Usage

- `RL_PPO_SB3.py`: Contains the RL training code and the environment setup. Modify the code to choose between sigmoid, SiLU, or GELU. After running code results are generated within the run_archive directory.
- `least_sq_approximation.py`: Solves the matrix equation and generates the piecewise linear functions.
- `points_to_function_converter.py`: Takes piecewise points to generate the variables that define the piecewise linear function.

Example usage of `points_to_function_converter.py`:
``````bash
python points_to_function_converter.py -F GELU -P -5 -4 0 1 -R -8 8 --round 5
``````

## Contributing

Contributions are welcome. Please read the contributing guidelines and code of conduct before making a pull request.

## Implementation Details

Takes inspiration from [Li et al 2022](https://doi.org/10.3390/electronics11091365) for using curvature to select piecewise linear points. We extend this by using RL (used PPO algorithm) for automated selection. The system leverages curvature as an intermediate reward and combines average and maximum error for the final reward.

Mathematical formulation:
The piecewise linear function is created given an input range, [b<sub>1</sub>, b<sub>m</sub>], and segment points (b<sub>2</sub>, ..., b<sub>m-1</sub>) using the least square method from [Li et al 2022](https://doi.org/10.3390/electronics11091365).
Range is set to [-8, 8] by default for all functions but can be modified if needed.

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/1ba53f1d-449c-478b-aaa8-52187889e19b)

Where A and B are left and right asymptotes respectively (e.g., for sigmoid A = 0, B = 1, and for SiLU and GELU A = 0, B = x). Given the range and segment points, the least square method is used to find the optimal beta values (&beta;<sub>1</sub>, &beta;<sub>2</sub>, ..., &beta;<sub>m</sub>) that minimizes the error, implemented using the scipy.linalg.lstsq function.

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/1b9925ae-2bfc-43c0-b7fc-6f3cfd85ed01)

The value x_i is sampled over given range [b<sub>1</sub>, b<sub>m</sub>] an over interval 0.001. So, for interval of [-8, 8] we have a total of 16001 points being sampled (as the range is inclusive). And value y_i is the output of a given function (sigmoid, SiLU, GELU, et cetera) when x_i is the input. The above matrix can be formulated into a least square problem as follows.

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/3d75a5df-0614-412a-be2d-501fe84f50c0)

The optimum <i>&beta;<sup>*</sup></i> that minimizes the error (<i>Y - A&beta;<sup>*</sup></i>) can be found using the least square method. In this implementation, it was done using the scipy.linalg.lstsq function.

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/2fcc1e55-cafa-44e6-8e01-825450e13e34)

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/ae59e192-a71f-400e-bdf4-c7b3b99334a9)


## Future Work

Code cleanup and reimplementation in Jax for performance improvement are in progress.

## Examples
- Curvatures of GELU, SiLU, and sigmoid functions:

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/de465649-2470-4321-a0e6-0e80b60df76d)

- Example of GELU, SiLU, and sigmoid functions with their piecewise linear counterparts:
  - GELU:
    ![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/f61686ea-56c7-469d-85ac-69b2d4c7cd28)
  - SiLU:
    ![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/9958a439-e49d-4c67-b8f2-fe26da86a825)
  - Sigmoid:
    ![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/be04b28b-f59c-417c-9750-5fea4c0c043c)
- ImageNet-1K Classification Accuracy using the approximation functions:
  ![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/daf20e43-ca9e-40f7-9057-3839c58e801b)

## License

TBD

## Contact Information

- **Author**: Kyumin Cho
- **Email**: kmcho@postech.ac.kr
- **Affiliation**: MS-PhD student at CAD & SoC Design Lab (CSDL), Postech under Professor Kang Seokhyeong

