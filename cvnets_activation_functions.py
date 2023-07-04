import torch
from torch import nn
from typing import Optional, Tuple

# @register_act_fn(name="swish_10_pwl_paper")

class Sigmoid(nn.Sigmoid):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__()

    def profile_module(self, input: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        return input, 0.0, 0.0

class Sigmoid10PWL(nn.Module):
    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        device = input.device  # Get the device of the input tensor
        zeros = torch.zeros_like(input, device=device)
        ones = torch.ones_like(input, device=device)

        return torch.where(
            input <= -8,
            zeros,
            torch.where(
                (input > -8) & (input <= -4.5),
                0.00252 * input + 0.01875,
                torch.where(
                    (input > -4.5) & (input <= -3),
                    0.02367 * input + 0.11397,
                    torch.where(
                        (input > -3) & (input <= -2),
                        0.06975 * input + 0.25219,
                        torch.where(
                            (input > -2) & (input <= -1),
                            0.14841 * input + 0.40951,
                            torch.where(
                                (input > -1) & (input <= 1),
                                0.2389 * input + 0.5,
                                torch.where(
                                    (input > 1) & (input <= 2),
                                    0.1481 * input + 0.59049,
                                    torch.where(
                                        (input > 2) & (input <= 3),
                                        0.06975 * input + 0.74781,
                                        torch.where(
                                            (input > 3) & (input <= 4.5),
                                            0.02367 * input + 0.88603,
                                            torch.where(
                                                (input > 4.5) & (input <= 8),
                                                0.00252 * input + 0.98125,
                                                ones
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

class Sigmoid10PWL_in_place(nn.Module):
    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        # Apply the piecewise-linear transformation in-place
        input.data.clamp_(-8, 8)

        mask1 = input <= -8
        input.mul_(0).masked_fill_(mask1, 0)

        mask2 = (input > -8) & (input <= -4.5)
        input.mul_(0.00252).add_(0.01875).masked_fill_(mask2, 0)

        mask3 = (input > -4.5) & (input <= -3)
        input.mul_(0.02367).add_(0.11397).masked_fill_(mask3, 0)

        mask4 = (input > -3) & (input <= -2)
        input.mul_(0.06975).add_(0.25219).masked_fill_(mask4, 0)

        mask5 = (input > -2) & (input <= -1)
        input.mul_(0.14841).add_(0.40951).masked_fill_(mask5, 0)

        mask6 = (input > -1) & (input <= 1)
        input.mul_(0.2389).add_(0.5).masked_fill_(mask6, 0)

        mask7 = (input > 1) & (input <= 2)
        input.mul_(0.1481).add_(0.59049).masked_fill_(mask7, 0)

        mask8 = (input > 2) & (input <= 3)
        input.mul_(0.06975).add_(0.74781).masked_fill_(mask8, 0)

        mask9 = (input > 3) & (input <= 4.5)
        input.mul_(0.02367).add_(0.88603).masked_fill_(mask9, 0)

        mask10 = (input > 4.5) & (input <= 8)
        input.mul_(0.00252).add_(0.98125).masked_fill_(mask10, 1)

        return input


class Swish10PWL(nn.Module):
    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        device = input.device  # Get the device of the input tensor
        zeros = torch.zeros_like(input, device=device)
        ones = torch.ones_like(input, device=device)

        return input * torch.where(
            input <= -8,
            zeros,
            torch.where(
                (input > -8) & (input <= -4.5),
                0.00252 * input + 0.01875,
                torch.where(
                    (input > -4.5) & (input <= -3),
                    0.02367 * input + 0.11397,
                    torch.where(
                        (input > -3) & (input <= -2),
                        0.06975 * input + 0.25219,
                        torch.where(
                            (input > -2) & (input <= -1),
                            0.14841 * input + 0.40951,
                            torch.where(
                                (input > -1) & (input <= 1),
                                0.2389 * input + 0.5,
                                torch.where(
                                    (input > 1) & (input <= 2),
                                    0.1481 * input + 0.59049,
                                    torch.where(
                                        (input > 2) & (input <= 3),
                                        0.06975 * input + 0.74781,
                                        torch.where(
                                            (input > 3) & (input <= 4.5),
                                            0.02367 * input + 0.88603,
                                            torch.where(
                                                (input > 4.5) & (input <= 8),
                                                0.00252 * input + 0.98125,
                                                ones
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

class Swish10PWL_in_place(nn.Module):
    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        device = input.device  # Get the device of the input tensor

        # Apply piecewise linear approximation inplace
        input.clamp_(-8, )
        input.mul_(torch.where(
            input <= -4.5,
            torch.tensor(0.00252, device=device) * input + 0.01875,
            torch.where(
                input <= -3,
                torch.tensor(0.02367, device=device) * input + 0.11397,
                torch.where(
                    input <= -2,
                    torch.tensor(0.06975, device=device) * input + 0.25219,
                    torch.where(
                        input <= -1,
                        torch.tensor(0.14841, device=device) * input + 0.40951,
                        torch.where(
                            input <= 1,
                            torch.tensor(0.2389, device=device) * input + 0.5,
                            torch.where(
                                input <= 2,
                                torch.tensor(0.1481, device=device) * input + 0.59049,
                                torch.where(
                                    input <= 3,
                                    torch.tensor(0.06975, device=device) * input + 0.74781,
                                    torch.where(
                                        input <= 4.5,
                                        torch.tensor(0.02367, device=device) * input + 0.88603,
                                        torch.where(
                                            input <= 8,
                                            torch.tensor(0.00252, device=device) * input + 0.98125,
                                            torch.ones_like(input, device=device)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ))

        return input

# class Swish10PWL_in_place(nn.Module):
#     def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
#         super().__init__()
#
#     def forward(self, input):
#         device = input.device  # Get the device of the input tensor
#
#         # Apply piecewise linear approximation inplace
#         input.clamp_(-8, 8)
#         input.mul_(torch.where(
#             input <= -4.5,
#             torch.tensor(0.00252, device=device) * input + 0.01875,
#             torch.where(
#                 input <= -3,
#                 torch.tensor(0.02367, device=device) * input + 0.11397,
#                 torch.where(
#                     input <= -2,
#                     torch.tensor(0.06975, device=device) * input + 0.25219,
#                     torch.where(
#                         input <= -1,
#                         torch.tensor(0.14841, device=device) * input + 0.40951,
#                         torch.where(
#                             input <= 1,
#                             torch.tensor(0.2389, device=device) * input + 0.5,
#                             torch.where(
#                                 input <= 2,
#                                 torch.tensor(0.1481, device=device) * input + 0.59049,
#                                 torch.where(
#                                     input <= 3,
#                                     torch.tensor(0.06975, device=device) * input + 0.74781,
#                                     torch.where(
#                                         input <= 4.5,
#                                         torch.tensor(0.02367, device=device) * input + 0.88603,
#                                         torch.tensor(0.00252, device=device) * input + 0.98125
#                                     )
#                                 )
#                             )
#                         )
#                     )
#                 )
#             )
#         ))
#
#         return input

# native SiLU function from PyTorch
class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        return input, 0.0, 0.0

in_place_function = Swish10PWL_in_place()
out_of_place_function = Swish10PWL()
native_function = Swish()

# compare the runtime speed of the layers on a large tensor
import timeit
# create an iteration loop to time the layers
def timeit_loop(layer, tensor, iterations=1000):
    # create a lambda function to call the layer
    timer = lambda: layer(tensor)
    # time the function
    time = timeit.timeit(timer, number=iterations)
    # return the time
    return time

# create a large tensor
test_tensor_size = (1000, 1000)#(1000, 1000)

tensor = torch.randn(test_tensor_size)

# a range of numbers in [-9,9] in 0.2 intervals
start_range = -9
end_range = 9
step = 0.2
output_check_tensor = torch.arange(start_range, end_range, step)

# time the in-place layer
tensor = torch.randn(test_tensor_size)
in_place_time = timeit_loop(in_place_function, tensor)
print(f"In-place time: {in_place_time}")

# time the out-of-place layer
tensor = torch.randn(test_tensor_size)
out_of_place_time = timeit_loop(out_of_place_function, tensor)
print(f"Out-of-place time: {out_of_place_time}")

# time the native layer
tensor = torch.randn(test_tensor_size)
native_time = timeit_loop(native_function, tensor)
print(f"Native time: {native_time}")

# check and compare the outputs
in_place_output = in_place_function(output_check_tensor)
output_check_tensor = torch.arange(start_range, end_range, step)
out_of_place_output = out_of_place_function(output_check_tensor)
output_check_tensor = torch.arange(start_range, end_range, step)
native_output = native_function(output_check_tensor)
output_check_tensor = torch.arange(start_range, end_range, step)

# print the outputs
print(f"In-place output: {in_place_output}")
print(f"Out-of-place output: {out_of_place_output}")
print(f"Native output: {native_output}")

# plot the outputs
import matplotlib.pyplot as plt
plt.plot(output_check_tensor, in_place_output, label="In-place")
plt.plot(output_check_tensor, out_of_place_output, label="Out-of-place")
plt.plot(output_check_tensor, native_output, label="Native")

plt.legend()
plt.show()

# compare the l2 norm of the outputs with native output being the reference
print(f"In-place l2 norm: {torch.norm(in_place_output - native_output)}")
print(f"Out-of-place l2 norm: {torch.norm(out_of_place_output - native_output)}")
print(f"Native l2 norm: {torch.norm(native_output - native_output)}")

# compare average difference between the outputs with native output being the reference
print(f"In-place average difference: {torch.mean(torch.abs(in_place_output - native_output))}")
print(f"Out-of-place average difference: {torch.mean(torch.abs(out_of_place_output - native_output))}")
print(f"Native average difference: {torch.mean(torch.abs(native_output - native_output))}")

# Same Procedure except with Sigmoid
native_sigmoid_function = Sigmoid()
approx_sigmoid_function = Sigmoid10PWL()
approx_in_place_sigmoid_function = Sigmoid10PWL_in_place()

# time the native layer
tensor = torch.randn(test_tensor_size)
approx_in_place_sigmoid_time = timeit_loop(approx_in_place_sigmoid_function, tensor)
print(f"Approx In-place Sigmoid time: {approx_in_place_sigmoid_time}")
tensor = torch.randn(test_tensor_size)
approx_sigmoid_time = timeit_loop(approx_sigmoid_function, tensor)
print(f"Out-of-place Sigmoid time: {approx_sigmoid_time}")
tensor = torch.randn(test_tensor_size)
native_sigmoid_time = timeit_loop(native_sigmoid_function, tensor)
print(f"Native Sigmoid time: {native_sigmoid_time}")



# plot the outputs
output_check_tensor = torch.arange(start_range, end_range, step)
native_sigmoid_output = native_sigmoid_function(output_check_tensor)
output_check_tensor = torch.arange(start_range, end_range, step)
approx_sigmoid_output = approx_sigmoid_function(output_check_tensor)
output_check_tensor = torch.arange(start_range, end_range, step)
plt.plot(output_check_tensor, native_sigmoid_output, label="Native Sigmoid")
plt.plot(output_check_tensor, approx_sigmoid_output, label="Approx Sigmoid")
plt.legend()
plt.show()










