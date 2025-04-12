package types

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

type EventType int

// TODO: 이벤트 확인
const (
	TokenCreatedV1 EventType = 1 + iota
	TokensBoughtV1
	TokensSoldV1
	PairCreatedV1
	LiquidityAddedV1
	TransferV1
)

func (e EventType) String() string {
	switch e {
	case TokenCreatedV1:
		return "TokenCreatedV1"
	case TokensBoughtV1:
		return "TokensBoughtV1"
	case TokensSoldV1:
		return "TokensSoldV1"
	case PairCreatedV1:
		return "PairCreatedV1"
	case LiquidityAddedV1:
		return "LiquidityAddedV1"
	case TransferV1:
		return "TransferV1"
	}

	return ""
}

var (
	TokenCreatedV1Hash = crypto.Keccak256Hash([]byte("TokenCreated(address,address,string,string)"))
	TokensBoughtV1Hash = crypto.Keccak256Hash([]byte("TokensBought(address,address,uint256,uint256,uint256,uint256,uint256,uint256)"))
	TokensSoldV1Hash   = crypto.Keccak256Hash([]byte("TokensSold(address,address,uint256,uint256,uint256,uint256,uint256,uint256)"))
	PairCreatedV1Hash  = crypto.Keccak256Hash([]byte("PairCreated(address,address,address,uint256)"))
	TransferV1Hash     = crypto.Keccak256Hash([]byte("Transfer(address,address,uint256)"))
)

// --> Topic이 추가 된다면, Topics와 TopicsMapper에 모두 추가해준다.
var Topics = []common.Hash{
	TokenCreatedV1Hash, TokensBoughtV1Hash, TokensSoldV1Hash, PairCreatedV1Hash, TransferV1Hash,
}

var TopicsMapper = map[common.Hash]EventType{
	TokenCreatedV1Hash: TokenCreatedV1,
	TokensBoughtV1Hash: TokensBoughtV1,
	TokensSoldV1Hash:   TokensSoldV1,
	PairCreatedV1Hash:  PairCreatedV1,
	TransferV1Hash:     TransferV1,
}
