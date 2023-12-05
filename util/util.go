package util

import (
	"fmt"
	"os"
	"strconv"
)

func WriteValuesToFile(values [][]float64, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	for i := 0; i < len(values); i++ {
		for j := 0; j < len(values[0]); j++ {
			f.WriteString(strconv.FormatFloat(values[i][j], 'f', 6, 64))
			f.WriteString(" ")
		}
	}
}
