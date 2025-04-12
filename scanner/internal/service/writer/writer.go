package writer

import (
	"encoding/json"
	"math/big"
	"os"

	"github.com/ethereum/go-ethereum"
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

		// TODO:
		hasPairCreated := false
		for _, l := range logs {
			eventType := TopicsMapper[l.Topics[0]]
			if eventType == PairCreated {
				commonEventData := EventData{
					BlockNumber: uint64(l.BlockNumber),
					TxHash:      l.TxHash,
					LogIndex:    l.Index,
					EventType:   eventType,
					Address:     common.BytesToHash(l.Address.Bytes()),
				}
				eventData := parseEventData(commonEventData, eventType, l.Topics, l.Data)
				if eventData == nil {
					w.log.Debug("Failed to parse event data", "eventType", eventType)
					continue
				}

				pairEvent := eventData.(PairCreatedEvent)
				global.GlobalVariable.AddFilterAddresses(pairEvent.Token0)
				global.GlobalVariable.AddFilterAddresses(pairEvent.Token1)
				global.GlobalVariable.AddFilterAddresses(pairEvent.Pair)

				hasPairCreated = true
			}
		}

		if hasPairCreated {
			startBlock := int64(logs[0].BlockNumber)
			endBlock := int64(logs[len(logs)-1].BlockNumber)

			filter := ethereum.FilterQuery{
				Addresses: global.GlobalVariable.GetFilterAddresses(),
				Topics:    [][]common.Hash{global.GlobalVariable.GetFilterTopics()},
				FromBlock: big.NewInt(startBlock),
				ToBlock:   big.NewInt(endBlock),
			}

			if newLogs, err := w.client.GetLogs(filter); err != nil {
				w.log.Error("Failed to get added filter logs", "func", "ConsumeClaim", "startBlock", startBlock, "endBlock", endBlock, "err", err)
			} else if len(newLogs) > 0 {
				logs = newLogs
			}
		}

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

			// 블록 타임스탬프 가져오기
			eventTimestamp, err := w.getBlockTimestamp(int64(l.BlockNumber))
			if err != nil {
				w.log.Error("Failed to get block timestamp", "error", err)
				continue
			}

			// 이벤트 데이터 파싱
			// 공통 EventData 설정
			commonEventData := EventData{
				BlockNumber: uint64(l.BlockNumber),
				TxHash:      l.TxHash,
				LogIndex:    l.Index,
				EventType:   eventType,
				Address:     common.BytesToHash(l.Address.Bytes()),
				Timestamp:   eventTimestamp,
			}

			eventData := parseEventData(commonEventData, eventType, l.Topics, l.Data)
			if eventData == nil {
				w.log.Debug("Failed to parse event data", "eventType", eventType)
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
		w.log.Error("Failed to get from address", "err", err, "txHash", log.TxHash.Hex())
		return nil, err
	}

	var v any
	if v, err = utils.ToJSON(event); err != nil {
		w.log.Error("Failed to convert to JSON", "err", err, "event", event)
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

		// JSON 파일에 이벤트 데이터 저장
		for _, model := range eventModels {
			updateModel, ok := model.(*mongo.UpdateOneModel)
			if !ok {
				w.log.Error("Failed to convert to UpdateOneModel")
				continue
			}

			// 모델에서 데이터 추출
			update, ok := updateModel.Update.(bson.M)
			if !ok {
				w.log.Error("Failed to convert update to bson.M")
				continue
			}

			// $setOnInsert에서 실제 데이터 추출
			eventData, ok := update["$setOnInsert"]
			if !ok {
				w.log.Error("Failed to get $setOnInsert data")
				continue
			}

			// 원본 Event 데이터를 직접 JSON 문자열로 변환
			jsonBytes, err := json.Marshal(eventData)
			if err != nil {
				w.log.Error("Failed to marshal event data to JSON", "err", err)
				continue
			}

			// 현재 작업 디렉토리 가져오기
			dir, err := os.Getwd()
			if err != nil {
				w.log.Error("Failed to get working directory", "err", err)
				dir = "."
			}

			// 파일명 생성 (절대 경로 + 체인명_이벤트로그.json)
			fileName := dir + "/" + w.chainName + "_eventlogs.json"
			w.log.Info("Writing event to JSON file", "file", fileName)

			// 파일에 JSON 데이터 추가 (Append 모드)
			if err := w.appendToJSONFile(fileName, jsonBytes); err != nil {
				w.log.Error("Failed to write to JSON file", "err", err)
			} else {
				w.log.Info("Event successfully written to JSON file")
			}
		}
	} else {
		w.log.Debug("No events to process for JSON file")
	}
}

// JSON 파일에 데이터를 추가하는 함수
func (w *Writer) appendToJSONFile(fileName string, data []byte) error {
	// 파일을 Append 모드로 열기 (없으면 생성)
	file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// JSON 데이터 끝에 줄바꿈 추가
	data = append(data, '\n')

	// 파일에 데이터 쓰기
	if _, err := file.Write(data); err != nil {
		return err
	}

	return nil
}
