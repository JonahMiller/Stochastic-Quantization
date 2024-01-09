import subprocess

def test(model, quant, prob_type=["linear"], e_type=["default"], test_name="default"):
    for i in model:
        for j in quant:
            for e in e_type:
                for p in prob_type:
                    subprocess.run(["python", "main.py", "--model", i, "--quant", j,
                                    "--prob_type", p, "--e_type", e, "--test_name", test_name])

def test2(model, quant, prob_type=["linear"], e_type=["default"], test_name="default"):
    for i in model:
        for j in quant:
            for e in e_type:
                for p in prob_type:
                    subprocess.run(["python", "inq_strats.py", "--model", i, "--quant", j,
                                    "--prob_type", p, "--e_type", e, "--test_name", test_name])


def table_1():
    name = "table_1"
    model = ["vgg9", "resnet20"]
    quant = ["fwn", "bwn", "sq_bwn_default_layer", "twn", "sq_twn_default_layer"]
    test(model, quant, test_name=name)

def table_3():
    name= "table_3"
    model = ["vgg9", "resnet20"]
    quant = ["sq_bwn_custom_layer", "sq_twn_custom_layer"]
    probs = ["constant", "linear", "sigmoid", "softmax"]
    es = ["default", "one_minus_invert"]
    test(model, quant, prob_type=probs, e_type=es, test_name=name)

def table_4():
    name = "table4"
    model = ["vgg9", "resnet20"]
    quant = ["sq_bwn_custom_filter", "sq_twn_custom_filter"]
    probs = ["sigmoid"]
    es = ["one_minus_invert"]
    test(model, quant, prob_type=probs, e_type=es, test_name=name)


if __name__ == "__main__":

    # table_1()

    # with open(f"txt_results/final.txt", 'a+') as f:
    #     f.write("#############################################################################################" + "\n")
    # f.close()

    # table_3()

    # with open(f"txt_results/final.txt", 'a+') as f:
    #     f.write("#############################################################################################" + "\n")
    # f.close()

    table_4()

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n")
    f.close()