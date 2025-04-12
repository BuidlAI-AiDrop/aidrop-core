package writer

import (
	"encoding/binary"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	. "github.com/rrabit42/aidrop-core/types"
	"github.com/shopspring/decimal"
)

type variable struct {
	chainName         string
	blockTimestampMap map[int64]int64
	serviceId         string
	networkId         *big.Int
}

func parseEventData(eventData EventData, eventType EventType, topics []common.Hash, data []byte) any {
	switch eventType {
	case Transfer:
		return TransferEvent{
			EventData: eventData,
			From:      common.BytesToAddress(topics[1][:]),
			To:        common.BytesToAddress(topics[2][:]),
			Value:     decimal.NewFromBigInt(new(big.Int).SetBytes(data[:32]), 0),
		}

	case Approval:
		return ApprovalEvent{
			EventData: eventData,
			Owner:     common.BytesToAddress(topics[1][:]),
			Spender:   common.BytesToAddress(topics[2][:]),
			Value:     decimal.NewFromBigInt(new(big.Int).SetBytes(data[:32]), 0),
		}

	case ApprovalForAll:
		return ApprovalForAllEvent{
			EventData: eventData,
			Owner:     common.BytesToAddress(topics[1][:]),
			Operator:  common.BytesToAddress(topics[2][:]),
			Approved:  new(big.Int).SetBytes(data[:32]).Uint64() == 1,
		}

	case TransferSingle:
		return TransferSingleEvent{
			EventData: eventData,
			Operator:  common.BytesToAddress(topics[1][:]),
			From:      common.BytesToAddress(topics[2][:]),
			To:        common.BytesToAddress(topics[3][:]),
			Id:        new(big.Int).SetBytes(data[:32]),
			Value:     new(big.Int).SetBytes(data[32:64]),
		}

	case TransferBatch:
		// 복잡한 배열 형태의 데이터는 실제 구현에서 추가 처리 필요
		// 간소화를 위해 배열은 다루지 않음
		return TransferBatchEvent{
			EventData: eventData,
			Operator:  common.BytesToAddress(topics[1][:]),
			From:      common.BytesToAddress(topics[2][:]),
			To:        common.BytesToAddress(topics[3][:]),
			// Ids와 Values 배열 파싱은 실제 구현에서 처리
		}

	case URI:
		// string 파싱은 실제 구현에서 추가 처리 필요
		return URIEvent{
			EventData: eventData,
			Id:        new(big.Int).SetBytes(data[32:64]),
		}

	case Swap:
		return SwapEvent{
			EventData:  eventData,
			Sender:     common.BytesToAddress(topics[1][:]),
			Amount0In:  new(big.Int).SetBytes(data[:32]),
			Amount1In:  new(big.Int).SetBytes(data[32:64]),
			Amount0Out: new(big.Int).SetBytes(data[64:96]),
			Amount1Out: new(big.Int).SetBytes(data[96:128]),
			To:         common.BytesToAddress(data[128:160]),
		}

	case PairCreated:
		return PairCreatedEvent{
			EventData: eventData,
			Token0:    common.BytesToAddress(topics[1][:]),
			Token1:    common.BytesToAddress(topics[2][:]),
			Pair:      common.BytesToAddress(data[:32]),
			PairCount: new(big.Int).SetBytes(data[32:64]).Int64(),
		}

	case Sync:
		return SyncEvent{
			EventData: eventData,
			Reserve0:  new(big.Int).SetBytes(data[:32]),
			Reserve1:  new(big.Int).SetBytes(data[32:64]),
		}

	case Mint:
		return MintEvent{
			EventData: eventData,
			To:        common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
		}

	case Burn:
		return BurnEvent{
			EventData: eventData,
			From:      common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
		}

	case Deposit:
		return DepositEvent{
			EventData:    eventData,
			User:         common.BytesToAddress(topics[1][:]),
			Reserve:      common.BytesToAddress(topics[2][:]),
			Amount:       new(big.Int).SetBytes(data[:32]),
			ReferralCode: uint16(new(big.Int).SetBytes(data[32:64]).Uint64()),
		}

	case Withdraw:
		return WithdrawEvent{
			EventData: eventData,
			User:      common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
			To:        common.BytesToAddress(data[32:64]),
		}

	case Borrow:
		return BorrowEvent{
			EventData:    eventData,
			User:         common.BytesToAddress(topics[1][:]),
			Reserve:      common.BytesToAddress(topics[2][:]),
			Amount:       new(big.Int).SetBytes(data[:32]),
			InterestRate: new(big.Int).SetBytes(data[32:64]),
			ReferralCode: uint16(new(big.Int).SetBytes(data[64:96]).Uint64()),
			BorrowRate:   new(big.Int).SetBytes(data[96:128]),
		}

	case Repay:
		return RepayEvent{
			EventData: eventData,
			User:      common.BytesToAddress(topics[1][:]),
			Repayer:   common.BytesToAddress(topics[2][:]),
			Reserve:   common.BytesToAddress(topics[3][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
			Interest:  new(big.Int).SetBytes(data[32:64]),
		}

	case LiquidationCall:
		return LiquidationCallEvent{
			EventData:        eventData,
			Collateral:       common.BytesToAddress(topics[1][:]),
			DebtAsset:        common.BytesToAddress(topics[2][:]),
			User:             common.BytesToAddress(topics[3][:]),
			DebtToCover:      new(big.Int).SetBytes(data[:32]),
			LiquidatedAmount: new(big.Int).SetBytes(data[32:64]),
			Liquidator:       common.BytesToAddress(data[64:96]),
			ReceiveAToken:    new(big.Int).SetBytes(data[96:128]).Uint64() == 1,
		}

	case NameRegistered:
		var labelHash [32]byte
		copy(labelHash[:], topics[1][:])
		return NameRegisteredEvent{
			EventData: eventData,
			LabelHash: labelHash,
			Owner:     common.BytesToAddress(topics[2][:]),
			Cost:      new(big.Int).SetBytes(data[:32]),
		}

	case NameRenewed:
		var labelHash [32]byte
		copy(labelHash[:], topics[1][:])
		return NameRenewedEvent{
			EventData: eventData,
			LabelHash: labelHash,
			Cost:      new(big.Int).SetBytes(data[:32]),
		}

	case NameTransferred:
		var labelHash [32]byte
		copy(labelHash[:], topics[1][:])
		return NameTransferredEvent{
			EventData: eventData,
			LabelHash: labelHash,
			Owner:     common.BytesToAddress(topics[2][:]),
		}

	case Staked:
		return StakedEvent{
			EventData: eventData,
			User:      common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
		}

	case Unstaked:
		return UnstakedEvent{
			EventData: eventData,
			User:      common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
		}

	case RewardClaimed:
		return RewardClaimedEvent{
			EventData: eventData,
			User:      common.BytesToAddress(topics[1][:]),
			Amount:    new(big.Int).SetBytes(data[:32]),
		}

	case VoteCast:
		return VoteCastEvent{
			EventData:  eventData,
			Voter:      common.BytesToAddress(topics[1][:]),
			ProposalId: new(big.Int).SetBytes(topics[2][:]),
			Support:    new(big.Int).SetBytes(data[:32]).Uint64() == 1,
			Weight:     new(big.Int).SetBytes(data[32:64]),
		}

	case ProposalCreated:
		return ProposalCreatedEvent{
			EventData:  eventData,
			ProposalId: new(big.Int).SetBytes(topics[1][:]),
			Proposer:   common.BytesToAddress(topics[2][:]),
			// Description 문자열 파싱은 실제 구현에서 처리
		}

	case OrderCreated:
		var orderId [32]byte
		copy(orderId[:], topics[1][:])
		return OrderCreatedEvent{
			EventData: eventData,
			OrderId:   orderId,
			Maker:     common.BytesToAddress(topics[2][:]),
			AssetAddr: common.BytesToAddress(data[:32]),
			TokenId:   new(big.Int).SetBytes(data[32:64]),
			Price:     new(big.Int).SetBytes(data[64:96]),
		}

	case OrderCancelled:
		var orderId [32]byte
		copy(orderId[:], topics[1][:])
		return OrderCancelledEvent{
			EventData: eventData,
			OrderId:   orderId,
		}

	case OrderMatched:
		var buyOrderId, sellOrderId [32]byte
		copy(buyOrderId[:], topics[1][:])
		copy(sellOrderId[:], topics[2][:])
		return OrderMatchedEvent{
			EventData:   eventData,
			BuyOrderId:  buyOrderId,
			SellOrderId: sellOrderId,
			Maker:       common.BytesToAddress(data[:32]),
			Taker:       common.BytesToAddress(data[32:64]),
			TokenId:     new(big.Int).SetBytes(data[64:96]),
			Price:       new(big.Int).SetBytes(data[96:128]),
		}

	case Paused:
		return PausedEvent{
			EventData: eventData,
			Account:   common.BytesToAddress(data[:32]),
		}

	case Unpaused:
		return UnpausedEvent{
			EventData: eventData,
			Account:   common.BytesToAddress(data[:32]),
		}

	case OwnershipTransferred:
		return OwnershipTransferredEvent{
			EventData:     eventData,
			PreviousOwner: common.BytesToAddress(topics[1][:]),
			NewOwner:      common.BytesToAddress(topics[2][:]),
		}

	case RoleGranted:
		var role [32]byte
		copy(role[:], topics[1][:])
		return RoleGrantedEvent{
			EventData: eventData,
			Role:      role,
			Account:   common.BytesToAddress(topics[2][:]),
			Sender:    common.BytesToAddress(topics[3][:]),
		}

	case RoleRevoked:
		var role [32]byte
		copy(role[:], topics[1][:])
		return RoleRevokedEvent{
			EventData: eventData,
			Role:      role,
			Account:   common.BytesToAddress(topics[2][:]),
			Sender:    common.BytesToAddress(topics[3][:]),
		}

	default:
		// 모르는 이벤트 타입의 경우 nil 반환
		return nil
	}
}

func genEventHash(log types.Log) common.Hash {
	// convert block number to bytes
	blockNumberBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(blockNumberBytes, log.BlockNumber)

	// convert tx index to bytes
	txIndexBytes := make([]byte, 4)
	binary.BigEndian.PutUint32(txIndexBytes, uint32(log.TxIndex))

	// convert log index to bytes
	indexBytes := make([]byte, 4)
	binary.BigEndian.PutUint32(indexBytes, uint32(log.Index))

	return crypto.Keccak256Hash(
		blockNumberBytes,
		log.TxHash[:],
		txIndexBytes,
		indexBytes,
	)
}
