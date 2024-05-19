package main

import (
	"Mitschl/Gomal/ml"
	//"Mitschl/Gomal/util"
	"fmt"
	//"math"
	"github.com/gin-gonic/gin"
)

type LearnRequestBody struct {
	Input  int   `json:"input"`
	Hidden []int `json:"hidden"`
	Output int   `json:"output"`

	X [][]float64 `json:"x"`
	Y [][]float64 `json:"y"`

	Iterations   int     `json:"iterations"`
	LearningRate float64 `json:"learningRate"`
}

func main() {

	r := gin.Default()

    r.LoadHTMLGlob("templ/*")

	r.POST("/train", func(c *gin.Context) {
		var requestBody LearnRequestBody
		err := c.BindJSON(&requestBody)

		if err != nil {
			fmt.Println(err.Error())
		}

		fmt.Println(requestBody.X, requestBody.Y, requestBody.Iterations, requestBody.LearningRate, requestBody.Input, requestBody.Hidden, requestBody.Output)



		W, B := ml.GradiantDescent(requestBody.X, requestBody.Y, requestBody.Iterations, requestBody.LearningRate, requestBody.Input, requestBody.Hidden, requestBody.Output)
		_, outputs := ml.ForwardPropMultiLayer(W, B, requestBody.X)

		c.JSON(200, outputs[len(outputs)-1])
	})

    r.GET("/list", func(c *gin.Context) {
        c.HTML(200, "list.templ", gin.H{"list": []int{1,2,3,4}})
    })

	r.Run() 

	/*
			X := make([][]float64, 1)
			X[0] = make([]float64, 70)
			for i := 0; i < 70; i++ {
				X[0][i] = float64(i) / 10.0
			}

		    fmt.Println(X)
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
