package chainRpc

import (
	"context"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	client "github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/rpc"
)

type Client struct {
	client    *client.Client
	signer    types.Signer
	networkID *big.Int
	url       string
}

func NewClient(url string) *Client {
	c := &Client{
		url: url,
	}

	rpc_, err := rpc.Dial(url)
	if err != nil {
		panic(err)
	}

	c.client = client.NewClient(rpc_)

	if c.networkID, err = c.client.NetworkID(context.Background()); err != nil {
		panic(err)
	}

	c.signer = types.NewEIP155Signer(c.networkID)

	return c
}

func (c *Client) GetNetworkID() *big.Int {
	return c.networkID
}

func (c *Client) GetBlockTimestamp(blockNumber int64) (int64, error) {
	if header, err := c.client.HeaderByNumber(context.Background(), big.NewInt(blockNumber)); err != nil {
		return 0, err
	} else {
		return int64(header.Time), nil
	}
}

func (c *Client) GetFromAddress(hash common.Hash) (common.Address, error) {
	var err error
	var tx *types.Transaction
	if tx, _, err = c.client.TransactionByHash(context.Background(), hash); err != nil {
		return common.Address{}, err
	}

	signer := types.NewEIP155Signer(tx.ChainId())
	return types.Sender(signer, tx)
}
