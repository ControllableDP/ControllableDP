Welcome to the Controllable DP-FL!

This is the code for Controllable DP-FL.


### Examples for **MNIST**
- MNIST
    ```
    cd ./dataset
    python generate_mnist.py iid - - # for iid and unbalanced scenario
    # python generate_mnist.py iid balance - # for iid and balanced scenario
    # python generate_mnist.py noniid - pat # for pathological noniid and unbalanced scenario
    # python generate_mnist.py noniid - dir # for practical noniid and unbalanced scenario
    ```



- Run evaluation: 
    ```bash
    cd ./system
    python3 main.py -data mnist -m -model dnn -algo FedDPCONTROL -gr 100 -sfn FedDPCONTROL -did 0 -ldp gaussian -crf True -Dclip True -kf True -crd CRD -pb 0.6 -t 10 # using the MNIST dataset, Gaussiam mechanism, the Fedavg + CONTROL algorithm

