# GoMaL

## What is GoMaL?

Gomal is a small machine learning library for feed-forward neural networks, as part of my university curriculum. The network is trained using gradient decent.

## Installation

### Prerequisits

#### Required

One of the following:

- go installed https://go.dev/doc/install
- Docker installed https://docs.docker.com/engine/install/

#### Optional

##### Visualisation

- python3 installed https://www.python.org/downloads/
- MatPlotLib installed https://matplotlib.org/stable/users/installing/index.html

## How to Run

### Using Go

When you are in the repository folder you can run:

```bash
go run main.go
```

to run the Programm.

### Using Docker

When you are in the repository folder you can run:

```bash
sudo docker build -t gomal .
```

to build a Docker-Image named gomal. After that you can use

```bash
sudo docker run gomal
```

to run the Image.

### Visualisation

Just run the Python script /util/helper.py in the repository folder. Be sure to install
