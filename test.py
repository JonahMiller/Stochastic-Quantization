import subprocess

def test(model, quant, prob_type=["linear"], e_type=["default"]):
    for i in model:
        for j in quant:
            for e in e_type:
                for p in prob_type:
                    subprocess.run(["python", "main.py", "--model", i, "--quant", j,
                                    "--prob_type", p, "--e_type", e])

def test2(model, quant, prob_type=["linear"], e_type=["default"]):
    for i in model:
        for j in quant:
            for e in e_type:
                for p in prob_type:
                    subprocess.run(["python", "inq_strats.py", "--model", i, "--quant", j,
                                    "--prob_type", p, "--e_type", e])


def table_1():
    model = ["vgg9", "resnet20"]
    quant = ["fwn", "bwn", "sq_bwn_default_layer", "twn", "sq_twn_default_layer"]
    test(model, quant)

def table_3():
    model = ["vgg9", "resnet20"]
    quant = ["sq_bwn_default_layer", "sq_twn_default_layer"]
    probs = ["constant", "linear", "sigmoid", "softmax"]
    test(model, quant, prob_type=probs)

def table_4():
    model = ["vgg9", "resnet20"]
    quant = ["sq_bwn_custom_layer", "sq_twn_custom_layer"]
    probs = ["constant", "linear", "sigmoid", "softmax"]
    es = ["default", "one_minus_invert"]
    test(model, quant, prob_type=probs, e_type=es)

def table_5():
    model = ["vgg9", "resnet20"]
    quant = ["sq_twn_custom_layer", "sq_twn_custom_layer"]
    probs = ["constant", "linear", "sigmoid", "softmax"]
    es = ["default", "one_minus_invert"]
    test(model, quant, prob_type=probs, e_type=es)


if __name__ == "__main__":
    table_1()

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n")
    f.close()

    table_3()

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n")
    f.close()

    table_4()

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n")
    f.close()

    table_5()

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n")
    f.close()