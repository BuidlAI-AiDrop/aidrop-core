package writer

import "math/big"

type variable struct {
	chainName         string
	blockTimestampMap map[int64]int64
	serviceId         string
	networkId         *big.Int
}
