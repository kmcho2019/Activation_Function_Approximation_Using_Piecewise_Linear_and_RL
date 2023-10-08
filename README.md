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

- `RL_PPO_SB3.py`: Contains the RL training code and the environment setup. Modify the code to choose between sigmoid, SiLU, or GELU.
- `least_sq_approximation.py`: Solves the matrix equation and generates the piecewise linear functions.
- `points_to_function_converter.py`: Takes piecewise points to generate the variables that define the piecewise linear function.

Example usage of `points_to_function_converter.py`:
``````bash
python points_to_function_converter.py -F GELU -P -5 -4 0 1 -R -8 8 --round 5
``````

## Contributing

Contributions are welcome. Please read the contributing guidelines and code of conduct before making a pull request.

## Implementation Details

Takes inspiration from [Li et al 2022](#citation) for using curvature to select piecewise linear points. We extend this by using RL (used PPO algorithm) for automated selection. The system leverages curvature as an intermediate reward and combines average and maximum error for the final reward.

Mathematical formulation:
![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/1ba53f1d-449c-478b-aaa8-52187889e19b)

Where A and B are left and right asymptotes respectively (e.g., for sigmoid A = 0, B = 1, and for SiLU and GELU A = 0, B = x). Given the range and segment points, the least square method is used to find the optimal beta values (\beta_1,\ \beta_2,\ \ldots\ \ ,\ \beta_m)\ that minimizes the error, implemented using the scipy.linalg.lstsq function.

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/1f0a144b-5549-4c32-954d-094c30579756)

![image](https://github.com/kmcho2019/Activation_Function_Approximation_Using_Piecewise_Linear_and_RL/assets/91612340/ec02bba7-8c42-4130-83df-1ac6157b7c77)


## Future Work

Code cleanup and reimplementation in Jax for performance improvement are in progress.

## Examples

- Example of GELU, SiLU, and sigmoid functions with their piecewise linear counterparts (insert visualizations here).

## License

TBD

## Contact Information

- **Author**: Kyumin Cho
- **Email**: kmcho@postech.ac.kr
- **Affiliation**: MS-PhD student at CAD & SoC Design Lab (CSDL), Postech under Professor Kang Seokhyeong

## Citations
``````bibtex
@article{li2022,
    title={FPGA implementation for the sigmoid with piecewise linear fitting method based on curvature analysis},
    author={Z. Li, Y. Zhang, B. Sui, Z. Xing, and Q. Wang},
    journal={Electronics},
    volume={11},
    number={9},
    pages={1365},
    year={2022},
    publisher={MDPI},
    doi={10.3390/electronics11091365}
}
``````
