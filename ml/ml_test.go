package ml

import (
	"fmt"
	"testing"
)

func compareMatrix(a [][]float64, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func TestSigmoid(t *testing.T) {
	var tests = []struct {
		input    float64
		expected float64
	}{
		{0, 0.5},
		{1, 0.7310585786300049},
		{-1, 0.2689414213699951},
	}
	for _, test := range tests {
		if output := sigmoid(test.input); output != test.expected {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestSigmoidMatrix(t *testing.T) {
	var tests = []struct {
		input    [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{0, 1, -1},
				{0, 1, -1},
			},
			[][]float64{
				{0.5, 0.7310585786300049, 0.2689414213699951},
				{0.5, 0.7310585786300049, 0.2689414213699951},
			},
		},
	}
	for _, test := range tests {
		if output := sigmoidMatrix(test.input); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestSigmoidDerivative(t *testing.T) {
	var tests = []struct {
		input    float64
		expected float64
	}{
		{0, 0.25},
		{1, 0.19661193324148185},
		{-1, 0.19661193324148185},
	}
	for _, test := range tests {
		if output := sigmoidDerivative(test.input); output != test.expected {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestSigmoidDerivativeMatrix(t *testing.T) {
	var tests = []struct {
		input    [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{0, 1, -1},
				{0, 1, -1},
			},
			[][]float64{
				{0.25, 0.19661193324148185, 0.19661193324148185},
				{0.25, 0.19661193324148185, 0.19661193324148185},
			},
		},
	}
	for _, test := range tests {
		if output := sigmoidDerivativeMatrix(test.input); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestMatrixMultiplication(t *testing.T) {
	var tests = []struct {
		inputA   [][]float64
		inputB   [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1},
				{2},
				{3},
				{4},
			},
			[][]float64{
				{1, 2, 3, 4, 5},
			},
			[][]float64{
				{1, 2, 3, 4, 5},
				{2, 4, 6, 8, 10},
				{3, 6, 9, 12, 15},
				{4, 8, 12, 16, 20},
			},
		},
	}
	for _, test := range tests {
		if output := matrixMultiplication(test.inputA, test.inputB); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.inputA, test.expected, output)
		}
	}
}

func TestMatrixAddVector(t *testing.T) {
	var tests = []struct {
		inputA   [][]float64
		inputB   [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 2, 3, 4, 5},
				{2, 4, 6, 8, 10},
				{3, 6, 9, 12, 15},
				{4, 8, 12, 16, 20},
			},
			[][]float64{
				{3},
				{2},
				{1},
				{4},
			},
			[][]float64{
				{4, 5, 6, 7, 8},
				{4, 6, 8, 10, 12},
				{4, 7, 10, 13, 16},
				{8, 12, 16, 20, 24},
			},
		},
	}
	for _, test := range tests {
		if output := matrixAddVector(test.inputA, test.inputB); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.inputA, test.expected, output)
		}
	}
}

func TestMatrixSum(t *testing.T) {
	var tests = []struct {
		input    [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			[][]float64{
				{6},
				{11},
				{16},
				{21},
			},
		},
	}
	for _, test := range tests {
		if output := matrixSum(test.input); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestMatrixScalarMultiplication(t *testing.T) {
	var tests = []struct {
		inputA   [][]float64
		inputB   float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			2,
			[][]float64{
				{2, 2, 2, 2, 4},
				{4, 4, 4, 4, 6},
				{6, 6, 6, 6, 8},
				{8, 8, 8, 8, 10},
			},
		},
	}
	for _, test := range tests {
		if output := matrixScalarMultiplication(test.inputA, test.inputB); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.inputA, test.expected, output)
		}
	}
}

func TestMatrixTranspose(t *testing.T) {
	var tests = []struct {
		input    [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 1, 1},
				{2, 2, 2},
				{3, 3, 3},
				{4, 4, 4},
			},
			[][]float64{
				{1, 2, 3, 4},
				{1, 2, 3, 4},
				{1, 2, 3, 4},
			},
		},
	}
	for _, test := range tests {
		if output := matrixTranspose(test.input); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.input, test.expected, output)
		}
	}
}

func TestMatrixSubstraction(t *testing.T) {
	var tests = []struct {
		inputA   [][]float64
		inputB   [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			[][]float64{
				{0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0},
			},
		},
	}
	for _, test := range tests {
		if output := matrixSubstraction(test.inputA, test.inputB); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.inputA, test.expected, output)
		}
	}
}

func TestMatrixMultiplicationByElement(t *testing.T) {
	var tests = []struct {
		inputA   [][]float64
		inputB   [][]float64
		expected [][]float64
	}{
		{
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			[][]float64{
				{1, 1, 1, 1, 2},
				{2, 2, 2, 2, 3},
				{3, 3, 3, 3, 4},
				{4, 4, 4, 4, 5},
			},
			[][]float64{
				{1, 1, 1, 1, 4},
				{4, 4, 4, 4, 9},
				{9, 9, 9, 9, 16},
				{16, 16, 16, 16, 25},
			},
		},
	}
	for _, test := range tests {
		if output := matrixMultiplicationByElement(test.inputA, test.inputB); !compareMatrix(output, test.expected) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.inputA, test.expected, output)
		}
	}

}

func TestForwardProp(t *testing.T) {
	var tests = []struct {
		W1 [][]float64
		W2 [][]float64
		b1 [][]float64
		b2 [][]float64
		Z1 [][]float64
		A1 [][]float64
		Z2 [][]float64
		A2 [][]float64
	}{
		{
			[][]float64{{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0}},
			[][]float64{{0.1, 0.2, -0.1, -0.2, 0.0}},
			[][]float64{{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0}},
			[][]float64{{0.1}},
			[][]float64{{0.2}, {0.4}, {-0.2}, {-0.4}, {0.0}},
			[][]float64{{0.549833997312478}, {0.5986876601124521}, {0.45016600268752216}, {0.401312339887548}, {0.5}},
			[][]float64{{0.1494418635074764}},
			[][]float64{{0.1494418635074764}},
		},
	}
	for _, test := range tests {
		outputZ1, outputA1, outputZ2, outputA2 := ForwardProp(test.W1, test.b1, test.W2, test.b2, [][]float64{{1}})
		fmt.Println("Z1", outputZ1)
		fmt.Println("A1", outputA1)
		fmt.Println("Z2", outputZ2)
		fmt.Println("A2", outputA2)
		if outputZ1, outputA1, outputZ2, outputA2 := ForwardProp(test.W1, test.b1, test.W2, test.b2, [][]float64{{1}}); !compareMatrix(outputZ1, test.Z1) || !compareMatrix(outputA1, test.A1) || !compareMatrix(outputZ2, test.Z2) || !compareMatrix(outputA2, test.A2) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.W1, test.Z1, outputZ1)
		}
	}
}


func TestUpdateParamsMultiLayer(t *testing.T) {
	var tests = []struct {
		weights   [][][]float64
		biases    [][][]float64
		dWeights  [][][]float64
		ddWeights [][][]float64
		dBiases   [][][]float64
		ddBiases  [][][]float64
		expectedW [][][]float64
		expectedB [][][]float64
	}{
		{
			[][][]float64{
				{
					{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0},
					{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0},
					{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0},
				},
				{
					{0.1, 0.2, -0.1, -0.2, 0.0},
					{0.1, 0.2, -0.1, -0.2, 0.0},
					{0.1, 0.2, -0.1, -0.2, 0.0},
				},
			},
			[][][]float64{
				{
					{0.1}, {0.2}, {-0.1}, {-0.2}, {0.0},
				},
				{
					{0.1},
				},
			},
			[][][]float64{
				{
					{0.004}, {0.008}, {-0.004}, {-0.008}, {0.0},
					{0.004}, {0.008}, {-0.004}, {-0.008}, {0.0},
					{0.004}, {0.008}, {-0.004}, {-0.008}, {0.0},
				},
				{
					{0.004, 0.008, -0.004, -0.008, 0.0},
					{0.004, 0.008, -0.004, -0.008, 0.0},
					{0.004, 0.008, -0.004, -0.008, 0.0},
				},
			},
			[][][]float64{
				{
					{0.0}, {0.0}, {0.0}, {0.0}, {0.0},
					{0.0}, {0.0}, {0.0}, {0.0}, {0.0},
					{0.0}, {0.0}, {0.0}, {0.0}, {0.0},
				},
				{
					{0.0, 0.0, 0.0, 0.0, 0.0},
					{0.0, 0.0, 0.0, 0.0, 0.0},
					{0.0, 0.0, 0.0, 0.0, 0.0},
				},
			},
			[][][]float64{
				{
					{0.004}, {0.008}, {-0.004}, {-0.008}, {0.0},
				},
				{
					{0.004},
				},
			},
			[][][]float64{
				{
					{0.0}, {0.0}, {0.0}, {0.0}, {0.0},
				},
				{
					{0.0},
				},
			},
			[][][]float64{
				{
					{0.0988}, {0.1976}, {-0.0988}, {-0.1976}, {0.0},
					{0.0988}, {0.1976}, {-0.0988}, {-0.1976}, {0.0},
					{0.0988}, {0.1976}, {-0.0988}, {-0.1976}, {0.0},
				},
				{
					{0.0988, 0.1976, -0.0988, -0.1976, 0.0},
					{0.0988, 0.1976, -0.0988, -0.1976, 0.0},
					{0.0988, 0.1976, -0.0988, -0.1976, 0.0},
				},
			},
			[][][]float64{
				{
					{0.0988}, {0.1976}, {-0.0988}, {-0.1976}, {0.0},
				},
				{
					{0.0988},
				},
			},
		},
	}
	for _, test := range tests {
		if outputW, outputB := updateParamsMultiLayer(test.weights, test.biases, test.dWeights, test.ddWeights, test.dBiases, test.ddBiases, 0.3, 0); !compareMatrix(outputW[0], test.expectedW[0]) || !compareMatrix(outputB[0], test.expectedB[0]) || !compareMatrix(outputW[1], test.expectedW[1]) || !compareMatrix(outputB[1], test.expectedB[1]) {
			t.Errorf("Test failed: input: %f, expected: %f, output: %f", test.weights, test.expectedW, outputW)
		}
	}
}

func TestRandomMatrix(t *testing.T) {
	var tests = []struct {
		rows    int
		columns int
	}{
		{5, 5},
		{5, 10},
		{10, 5},
		{10, 10},
	}
	for _, test := range tests {
		if output := randomMatrix(test.rows, test.columns, 0.5, -0.5); len(output) != test.rows || len(output[0]) != test.columns {
			t.Errorf("Test failed: input: %d, %d, expected: %d, %d, output: %d, %d", test.rows, test.columns, test.rows, test.columns, len(output), len(output[0]))
		}
	}
}
