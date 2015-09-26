package main 

//

import (
    "os"
    "fmt"
	"io"
	"encoding/binary"
)

//

//read mnist labels from byte file
func DataLoadLabels(r io.Reader) ([]byte) {
    header := [2]int32{}
    binary.Read(r, binary.BigEndian, &header)
    labels := make([]byte, header[1])
    r.Read(labels)
    return labels
}

//read mnist images from byte file
func DataLoadImages(r io.Reader) ([][]byte, int, int) {
    header := [4]int32{}
    binary.Read(r, binary.BigEndian, &header)
    images := make([][]byte, header[1])
    width, height := int(header[2]), int(header[3])
    for i := 0; i < len(images); i++ {
        images[i] = make([]byte, width * height)
        r.Read(images[i])
    }
    return images, width, height
}

func OpenFile( path string ) *os.File {
    file, err := os.Open(path)
    if (err != nil) {
        fmt.Println(err)
        os.Exit(-1)
    }
    return file
}

func PreprocessGlobalContrastStandardisation( M [][]byte, scalar float64 ) [][]float64 {
    // Needs to be rewritten to take global mean.
    fmt.Println("\tPreprocessing data: Global Contrast Normalisation")
    rows := len(M)
    output := make([][]float64,rows)
    for i:=0; i<rows; i++ {
        output[i] = make([]float64,len(M[i]))
        for j:=0; j<len(M[i]); j++ {
            output[i][j] = float64(M[i][j]) / scalar
        }
    }
    return output
}

func PreprocessLabelsToVector( labels []byte, nClasses int ) [][]float64 {
    // Labels is a vector holding the actual digit-labels.
    // this is specifically for the MNIST data
    nRows := len(labels)
    output := make([][]float64,nRows)
    for i:=0; i<nRows; i++ {
        //create the array for the array 'result' to hold
        tmp := make([]float64,nClasses)
        // Now set the actual index to 1.
        if labels[i] == 10 { // Backup in case you have the dataset where 0 are encoded by 10s.
            tmp[0] = 1.0
        } else {
            tmp[labels[i]] = 1.0
        }
        output[i] = tmp
    }
    return output
}
