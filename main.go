package main

import (
	"Mitschl/Gomal/ml"
	"Mitschl/Gomal/util"
	"math"
)

func main() {
	X := make([][]float64, 1)
	X[0] = make([]float64, 70)
	for i := 0; i < 70; i++ {
		X[0][i] = float64(i) / 10.0
	}

	Y := make([][]float64, 1)
	Y[0] = make([]float64, 70)
	for i := 0; i < 70; i++ {
		Y[0][i] = math.Sin(float64(i) / 10.0)
	}

	util.WriteValuesToFile(Y, "./data/Ref.txt")

	W1, b1, W2, b2 := ml.GradiantDescent(X, Y, 600000, 0.003, 1, []int{7}, 1)

	_, _, _, A2 := ml.ForwardProp(W1, b1, W2, b2, X)

	util.WriteValuesToFile(A2, "./data/OutFin.txt")

	/*
		// test data for XOR
		X := [][]float64{
			{0, 0, 1, 1},
			{0, 1, 0, 1},
		}

		Y := [][]float64{
			{0, 1, 1, 0},
		}

		util.WriteValuesToFile(Y, "./data/Ref.txt")

		W1, b1, W2, b2 := ml.GradiantDescent(X, Y, 50000, 0.01, 2, []int{5}, 1)

		_, _, _, A2 := ml.ForwardProp(W1, b1, W2, b2, X)

		util.WriteValuesToFile(A2, "./data/OutFin.txt")
	*/
}
