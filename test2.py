import subprocess

prob_type = ["constant", "linear", "softmax", "sigmoid"]
e_type = ["one_minus_invert", "default"]


def test(model, quant, prob_type=["linear"], e_type=["default"], test_name="default"):
    for i in model:
        for j in quant:
            for e in e_type:
                for p in prob_type:
                    subprocess.run(["python", "main2.py", "--model", i, "--quant", j,
                                    "--prob_type", p, "--e_type", e, "--test_name", test_name])
            
def table_5():
    name = "table_5"
    model = ["vgg9", "resnet20"]
    quant = ["inq", "elq", "sq_elq"]
    probs = ["sigmoid"]
    es = ["one_minus_invert"]
    test(model, quant, prob_type=probs, e_type=es, test_name=name)
            
if __name__ == "__main__":
    test()