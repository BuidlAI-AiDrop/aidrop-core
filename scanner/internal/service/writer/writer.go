package writer

import (
	"math/big"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	"github.com/rrabit42/aidrop-core/internal/service/chainRpc"
)

type Writer struct {
	client *chainRpc.Client
	log    log.Logger
	*variable
}

func NewWriter(
	rpcClient *chainRpc.Client,
	networkId *big.Int,
	chainName string,
) *Writer {
	w := &Writer{
		client: rpcClient,
		variable: &variable{
			chainName:         chainName,
			blockTimestampMap: make(map[int64]int64),
			networkId:         networkId,
		},
		log: log.New("module", "service/writer"),
	}
	return w
}

func (w *Writer) Run(logsC <-chan []types.Log) {
	for logs := range logsC {
		// for global.GlobalVariable.FetchSubAvgState.Load() {
		// 	time.Sleep(1e9)
		// }

		var (
		// eventModels []mongo.WriteModel
		// tradeModels []mongo.WriteModel
		)

		w.log.Info("Claim block writer", "start", logs[0].BlockNumber, "end", logs[len(logs)-1].BlockNumber, "lenLogs", len(logs))

		// var txMappingLogs = make(map[common.Hash]types.Log)

		// for _, l := range logs {
		// 	txHash := l.TxHash
		// 	if _, ok := txMappingLogs[txHash]; !ok {
		// 		txMappingLogs[txHash] = make([]types.Log, 0)
		// 	}
		// 	txMappingLogs[txHash] = append(txMappingLogs[txHash], l)

		// 	eventType := TopicsMapper[l.Topics[0]]
		// 	eventData := parseEventData(eventType, l.Topics, l.Data)
		// }

	}

}
