import sys
from collections.abc import Callable,Iterable
from typing import Optional
import torch
import math


def problem_1a():
    print(chr(0))


def problem_1b():
    print(chr(0).__repr__())


def problem_1c():
    print("Testing chr(0):")
    print("chr(0) =", chr(0))
    print("print(chr(0)) =", end=" ")
    print(chr(0))
    print()
    
    test_string = "this is a test" + chr(0) + "string"
    print("String with chr(0):", repr(test_string))
    print("print() of that string:", end=" ")
    print(test_string)


def problem_2b():

    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    try:
        word = "café"
        print(decode_utf8_bytes_to_str_wrong(word.encode("utf-8")))
    except Exception as e:
        print("Found a Word that cannot be decoded: ", word)

def problem_2c():

    two_bytes=b"\x80\x80"

    try:
        print(two_bytes.decode("utf-8"))
    except Exception as e:
        print("Found a two byte sequence that cannot be decoded: ", two_bytes)

def problem_learning_rate_tuning(lr):

    class SGD(torch.optim.Optimizer):

        def __init__(self,params,lr=1e-3):
            if lr < 0:
                raise ValueError(f"Invalid learning rate:{lr}")
            defaults = {"lr":lr}
            super().__init__(params,defaults)

        def step(self,closure:Optional[Callable] = None):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    t = state.get("t",0)
                    grad = p.grad.data
                    p.data -= lr/math.sqrt(t+1) * grad
                    state["t"] = t+1
        
            return loss

    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights],lr = lr)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

if __name__ == "__main__":

    if sys.argv[1] == "1a":
        problem_1a()

    elif sys.argv[1] == "1b":
        problem_1b()

    elif sys.argv[1] == "1c":
        problem_1c()

    elif sys.argv[1] == "2b":
        problem_2b()

    elif sys.argv[1] == "2c":
        problem_2c()

    elif sys.argv[1] == "problem_learning_rate_tuning":
        problem_learning_rate_tuning(lr=1)

