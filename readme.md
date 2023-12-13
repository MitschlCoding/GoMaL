# GoMaL

## What is GoMaL?

Gomal is a small machine learning library for feed-forward neural networks, as part of my university curriculum. The network is trained using gradient decent.

## Installation

### Prerequisits

#### Required

- go installed https://go.dev/doc/install or Docker installed

#### Optional

- python3 installed for visualisation https://www.python.org/downloads/

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
