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
            
def table_4():
    name = "table_4"
    model = ["vgg9", "resnet20"]
    quant = ["inq", "elq"]
    test(model, quant, test_name=name)
            
def table_5():
    name = "table_5_2"
    model = ["vgg9"]
    quant = ["sq_elq_custom_filter", "sq_elq_custom_layer"]
    probs = ["constant", "linear", "softmax", "sigmoid"]
    es = ["one_minus_invert", "default"]
    test(model, quant, prob_type=probs, e_type=es, test_name=name)
            
if __name__ == "__main__":
    # table_4()

    # with open(f"txt_results/final2.txt", 'a+') as f:
    #     f.write("#############################################################################################" + "\n\n")
    # f.close()

    table_5()

    with open(f"txt_results/final2.txt", 'a+') as f:
        f.write("#############################################################################################" + "\n\n")
    f.close()