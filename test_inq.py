import subprocess

prob_type = ["constant", "linear", "softmax", "sigmoid"]
e_type = ["one_minus_invert", "default"]

def test():
    for e in e_type:
        for p in prob_type:
            subprocess.run(["python", "inq_strats.py", "--quant", "sq_elq", "--dataset", "cifar10",
                            "--prob_type", f"{p}", "--e_type", f"{e}"])
            
if __name__ == "__main__":
    test()