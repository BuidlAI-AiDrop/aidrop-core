package writer

import (
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	"github.com/rrabit42/aidrop-core/global"
	"github.com/rrabit42/aidrop-core/internal/service/chainRpc"
	"github.com/rrabit42/aidrop-core/internal/service/utils"
	. "github.com/rrabit42/aidrop-core/types"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
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
			eventModels []mongo.WriteModel
		)

		w.log.Info("Claim block writer", "start", logs[0].BlockNumber, "end", logs[len(logs)-1].BlockNumber, "lenLogs", len(logs))

		var txMappingLogs = make(map[common.Hash][]types.Log)

		for _, l := range logs {
			txHash := l.TxHash
			if _, ok := txMappingLogs[txHash]; !ok {
				txMappingLogs[txHash] = make([]types.Log, 0)
			}
			txMappingLogs[txHash] = append(txMappingLogs[txHash], l)

			// 이벤트 타입 확인
			eventType, ok := TopicsMapper[l.Topics[0]]
			if !ok {
				w.log.Debug("Unknown event topic", "topic", l.Topics[0].Hex())
				continue
			}

			// 이벤트 데이터 파싱
			eventData := parseEventData(eventType, l.Topics, l.Data)
			if eventData == nil {
				w.log.Debug("Failed to parse event data", "eventType", eventType)
				continue
			}

			// 블록 타임스탬프 가져오기
			eventTimestamp, err := w.getBlockTimestamp(int64(l.BlockNumber))
			if err != nil {
				w.log.Error("Failed to get block timestamp", "error", err)
				continue
			}

			eventModel, err := w.makeUpsertEventModel(l, eventType, eventData, eventTimestamp)
			if err != nil {
				w.log.Error("Failed to make upsert event model", "error", err)
				continue
			}
			eventModels = append(eventModels, eventModel)
		}

		// 이벤트 데이터베이스에 저장
		if len(eventModels) > 0 {
			w.eventsToDB(eventModels)

			global.GlobalVariable.LatestReceivedBlockNumber.Store(logs[len(logs)-1].BlockNumber)
			w.log.Info("Block write complete", "start", logs[0].BlockNumber, "write", logs[len(logs)-1].BlockNumber)
		}
	}
}

func (w *Writer) makeUpsertEventModel(log types.Log, eventType EventType, eventData any, eventTime int64) (mongo.WriteModel, error) {
	event := Event{
		EventHash:   genEventHash(log),
		NetworkId:   w.networkId,
		BlockNumber: int64(log.BlockNumber),
		TxHash:      log.TxHash,
		ToAddress:   log.Address,
		Index:       log.Index,
		EventType:   eventType,
		EventName:   eventType.String(),
		EventData:   eventData,
		EventTime:   eventTime,
	}

	var err error
	if event.FromAddress, err = w.client.GetFromAddress(log.TxHash); err != nil {
		return nil, err
	}

	var v any
	if v, err = utils.ToJSON(event); err != nil {
		return nil, err
	} else {
		return mongo.NewUpdateOneModel().
			SetFilter(bson.M{"event_hash": event.EventHash}).
			SetUpdate(bson.M{"$setOnInsert": v}).
			SetUpsert(true), nil
	}
}

func (w *Writer) getBlockTimestamp(blockNumber int64) (int64, error) {
	if _, ok := w.blockTimestampMap[blockNumber]; !ok {
		// delete previous block timestamp
		w.blockTimestampMap = make(map[int64]int64)
		if timestamp, err := w.client.GetBlockTimestamp(blockNumber); err != nil {
			return 0, err
		} else {
			w.blockTimestampMap[blockNumber] = timestamp
		}
	}
	return w.blockTimestampMap[blockNumber], nil
}

func (w *Writer) eventsToDB(eventModels []mongo.WriteModel) {
	if len(eventModels) > 0 {
		// if err := w.mongoDB.WememeDB.BulkWrite(eventModels, w.mongoDB.WememeDB.OrganizerCollection()); err != nil {
		// 	w.log.Error("Failed to bulk write events", "err", err)
		// }
	}
}
