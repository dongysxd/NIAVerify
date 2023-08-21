NIAVerify
=========

# Description

---

NIAVerify is a complete verification tool for Relu-based feed-forward neural networks. NIAVerify implements a verification method based on Neuron Importance, which measures the influence of neurons in the network on the final objective function and calculates its importance. The most important neurons are split to refine the constraints, thus transforming the original verification problem into a set of subproblems to solve. The subproblem has a smaller solution space than the original problem.

# Requirements

---

* Python 3.9 or higher
* Gurobi 9.5 or higher.

# Flow Chart

![flowchart.jpg](https://x.imgs.ovh/x/2023/08/20/64e208137552c.jpg)

# Installation

---

## Install Gurobi

```sh
wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
tar xvfz gurobi9.5.1_linux64.tar.gz
```

### Add the following to the .bashrc file:

```sh
export GUROBI_HOME="Current_directory/gurobi951/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

### Retrieve a Gurobi License

To run Gurobi one needs to obtain license from [here](https://www.gurobi.com/documentation/9.5/quickstart_linux/retrieving_and_setting_up_.html#section:RetrieveLicense).

## Install NIAVerify

```
pipenv install
pipenv shell
```

# Usage

---

```sh
python3     ./niaverify/tests/verification/mnistfc.py 
  
```

# Contributors

---

* Yansong Dong (lead contact) - ysdong@stu.xidian.edu.cn
* Yuehao Liu - isliuyuehao@163.com

# License and Copyright

---

Licensed under the [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause)
