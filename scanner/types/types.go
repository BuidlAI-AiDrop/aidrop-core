package types

import (
	"math/big"

	"github.com/ethereum/go-ethereum/common"
)

type Event struct {
	EventHash   common.Hash    `json:"event_hash" bson:"event_hash"`
	NetworkId   *big.Int       `json:"network_id"`
	BlockNumber int64          `json:"block_number" bson:"block_number"`
	TxHash      common.Hash    `json:"tx_hash" bson:"tx_hash"`
	FromAddress common.Address `json:"from_address" bson:"from_address"`
	ToAddress   common.Address `json:"to_address" bson:"to_address"`
	Index       uint           `json:"index" bson:"index"`
	EventType   EventType      `json:"event_type" bson:"event_type"`
	EventName   string         `json:"event_name" bson:"event_name"`
	EventData   any            `json:"event_data" bson:"event_data"`
	EventTime   int64          `json:"event_time" bson:"event_time"`
}
